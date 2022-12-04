import json

import torch
import torch.nn as nn

def load_json(f_name, type):
    with open("../../cfgs/"+type+"/"+f_name) as f:
        cfg_f= f.read()
    return json.loads(cfg_f)

def mod_resnet(resnet_model):
    resnet_model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=2, bias=False
    )
    resnet_model.maxpool = nn.Identity()
    return resnet_model

def set_requires_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val

def load_model_state(model, ckpt_path, key="state_dict", freeze_encoder=True):
    # 'state_dict' = key for pre-trained models, 'model' for FB Imagenet
    state = torch.load(ckpt_path)[key] 
    for k in list(state.keys()):
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt_path}")
    if freeze_encoder:
        set_requires_grad(model, False)
    return model

def list_all_equal(list):
    first = list[0]
    return all(first == x for x in list)
