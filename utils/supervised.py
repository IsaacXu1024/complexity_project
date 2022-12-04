# Code partially taken and modified from Solo-Learn module by vturrisi under MIT License

import logging
import omegaconf

import pytorch_lightning as pl
import solo.backbones as backbones
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from solo.utils.lars import LARS
from solo.utils.metrics import accuracy_at_k
from solo.utils.misc import omegaconf_select, remove_bias_and_norm_from_weight_decay
from torch.optim.lr_scheduler import MultiStepLR

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from utils.utils import load_model_state, mod_resnet

def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs

class SupervisedMethod(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]
    
    _BACKBONES = {
        "vit_small_mae": backbones.vit.vit_mae.vit_small,
        "vit_small_mocov3": backbones.vit.vit_mocov3.vit_small,
        "resnet18": backbones.resnet.resnet18 
    }

    def __init__(self, ckpt_path, cfg: omegaconf.DictConfig):
        super().__init__()

        cfg = self.add_and_assert_specific_cfg(cfg)
        
        self.cfg: omegaconf.DictConfig = cfg

        # Backbone
        self.backbone_args: Dict[str, Any] = cfg.backbone.kwargs

        kwargs = self.backbone_args.copy()
        
        backbone_arch: str = cfg.backbone.name 
        dataset: str = cfg.data.dataset
        method: str = cfg.method
        
        if "vit" in backbone_arch:
            self.backbone = self._BACKBONES[backbone_arch+"_"+method](**kwargs)
            self.features_dim: int = self.backbone.num_features
        if "resnet" in backbone_arch:
            self.backbone = self._BACKBONES[backbone_arch](method, **kwargs)
            self.features_dim: int = self.backbone.inplanes
            self.backbone.fc = nn.Identity()
            if "cifar" in dataset:
                self.backbone = mod_resnet(self.backbone)
        self.backbone = load_model_state(self.backbone, ckpt_path)
    
        # Classifier
        self.num_classes: int = cfg.data.num_classes
        self.classifier: nn.Module = nn.Sequential(
            nn.BatchNorm1d(self.features_dim, affine=False),
            nn.Linear(self.features_dim, self.num_classes)
        )

        # training related
        self.max_epochs: int = cfg.max_epochs
        self.accumulate_grad_batches: Union[int, None] = cfg.accumulate_grad_batches

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs
        self.exclude_bias_n_norm_wd: bool = cfg.optimizer.exclude_bias_n_norm_wd

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval
        assert self.scheduler_interval in ["step", "epoch"]
        if self.scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # if accumulating gradient then scale lr
        if self.accumulate_grad_batches:
            self.lr = self.lr * self.accumulate_grad_batches
            self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
            self.min_lr = self.min_lr * self.accumulate_grad_batches
            self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

        # for performance
        self.no_channel_last = cfg.performance.disable_channel_last

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.
        Args:
            cfg (omegaconf.DictConfig): DictConfig object.
        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        # default for extra backbone kwargs (use pytorch's default if not available)
        cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})

        # default parameters for optimizer
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(
            cfg, "optimizer.exclude_bias_n_norm_wd", False
        )
        # default for extra optimizer kwargs (use pytorch's default if not available)
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})

        # default for acc grad batches
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", None)

        # default parameters for the scheduler
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")

        # default parameters for performance optimization
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", False
        )

        return cfg

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.
        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.
        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        learnable_params = self.learnable_params

        # exclude bias and norm from weight decay
        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        # indexes of parameters without lr scheduler
        idxs_no_scheduler = [i for i, m in enumerate(learnable_params) if m.pop("static_lr", False)]

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        # create optimizer
        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        if self.scheduler.lower() == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler["scheduler"].get_lr
                if isinstance(scheduler, dict)
                else scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            if isinstance(scheduler, dict):
                scheduler["scheduler"].get_lr = partial_fn
            else:
                scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

    def forward(self, X) -> Dict:
        """Basic forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.
        Args:
            X (torch.Tensor): batch of images in tensor format.
        Returns:
            Dict: dict of logits and features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        logits = self.classifier(feats)
        return {"logits": logits, "feats": feats}
        
    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.
        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.
        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self(X)
        logits = out["logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # handle when the number of classes is smaller than 5
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        out.update({"loss": loss, "acc1": acc1, "acc5": acc5})
        return out

    def base_training_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Allows user to re-write how the forward step behaves for the training_step.
        Should always return a dict containing, at least, "loss", "acc1" and "acc5".
        Defaults to _base_shared_step
        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.
        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        return self._base_shared_step(X, targets)

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        _, X, targets = batch

        outs = [self.base_training_step(X, targets)]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        outs["loss"] = sum(outs["loss"])
        outs["acc1"] = sum(outs["acc1"])
        outs["acc5"] = sum(outs["acc5"])

        metrics = {
            "train_class_loss": outs["loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return outs

    def test_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        _, X, targets = batch

        out = self(X)
        logits = out["logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)

        labels_hat = torch.argmax(logits, dim=1)
        test_acc = torch.sum(targets == labels_hat).item() / (len(targets) * 1.0)

        self.log_dict({'test_loss': loss, 'test_acc': test_acc}, sync_dist=True)

        return out