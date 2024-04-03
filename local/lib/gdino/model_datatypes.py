#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
from torch import Tensor

from dataclasses import dataclass


# ---------------------------------------------------------------------------------------------------------------------
#%% Data types

@dataclass
class TextEncoding:
    
    ''' Data type representing the output of the text encoder '''
    
    text_prompt: str
    features: Tensor
    token_mask: Tensor
    attn_mask: Tensor
    position_ids: Tensor
    input_ids: Tensor
    idx_to_text_lookup: list[str]

    def to(self, device, dtype = None):
        self.features = self.features.to(device, dtype)
        self.token_mask = self.token_mask.to(device, torch.bool)
        self.attn_mask = self.attn_mask.to(device, torch.bool)
        self.position_ids = self.position_ids.to(device, torch.int32)
        self.input_ids = self.input_ids.to(device, torch.int32)
        return self


@dataclass
class ImageEncoding:
    
    ''' Data type representing the output of the image encoder '''
    
    features_list: list[Tensor]
    posencs_list: list[Tensor]
    masks_list: list[Tensor]
    
    def __len__(self):
        return len(self.features_list)
    
    def to(self, device, dtype = None):
        self.features_list = [item.to(device, dtype) for item in self.features_list]
        self.posencs_list = [item.to(device, dtype) for item in self.posencs_list]
        self.masks_list = [item.to(device, torch.bool) for item in self.masks_list]
        return self


@dataclass
class RawPredictions:
    
    ''' Data type representing the raw output of the text-to-image decoder '''
    
    logits: Tensor
    boxes_xywh_norm: Tensor
    
    def __len__(self):
        is_batched = self.logits.ndim == 3
        return self.logits.shape[1] if is_batched else len(self.logits)
    
    def to(self, device, dtype = None):
        self.logits = self.logits.to(device, dtype)
        self.boxes_xywh_norm = self.boxes_xywh_norm.to(device, dtype)
        return self
    
    def cpu(self):
        return self.to("cpu", torch.float32)


@dataclass
class DetectionResults:
    
    ''' Data type representing the final/filtered detection results '''
    
    text_prompt: str
    boxes_xywh_norm: Tensor
    labels: list[str]
    scores: Tensor
    have_detections: bool
    
    def __len__(self):
        return len(self.boxes_xywh_norm)
    
    def to(self, device, dtype = None):
        self.boxes_xywh_norm = self.boxes_xywh_norm.to(device, dtype)
        self.scores = self.scores.to(device, dtype)
        return self
    
    def cpu(self):
        return self.to("cpu", torch.float32)
    
    def as_dict(self):
        return {
            "text_prompt": self.text_prompt,
            "boxes_xywh_norm": self.boxes_xywh_norm.tolist(),
            "labels": self.labels,
            "scores": self.scores.tolist(),
        }
