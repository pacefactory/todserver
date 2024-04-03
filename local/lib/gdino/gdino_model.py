#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn

# Typing
from local.lib.gdino.text_processing import TextEncoder
from local.lib.gdino.image_processing import ImageEncoder
from local.lib.gdino.text_image_transformer import TextAndImageTransformer
from .model_datatypes import TextEncoding, ImageEncoding, RawPredictions, DetectionResults
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class GDinoModel(nn.Module):
    
    '''
    Wrapper around the model components of the 'Grounding DINO' model from:
        "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
        By: Shilong Liu, Zhaoyang Zeng, Tianhe Ren, et al.
        @ https://arxiv.org/abs/2303.05499
        @ https://github.com/IDEA-Research/GroundingDINO
    
    This version is slightly simplified from the original (mostly to remove dependencies)
    and has support for caching of text & image encodings, for quicker re-use.
    '''
    
    # .................................................................................................................
    
    def __init__(self,
                 text_encoder_model: TextEncoder,
                 image_encoder_model: ImageEncoder,
                 cross_encoder_model: TextAndImageTransformer):
        
        # Inherit from parent
        super().__init__()
        
        # Set up main model components
        self.txtenc_model = text_encoder_model
        self.imgenc_model = image_encoder_model
        self.crossenc_model = cross_encoder_model
        
        # Set up encoder caches for re-use
        self.cache_txtenc = DummyCache()
        self.cache_imgenc = DummyCache()
    
    # .................................................................................................................
    
    def unload_resources(self):
        
        '''
        Clear references to models/data to help with cleanup of resources
        Note, this leaves the model in a unusable state! The class instance is expected
        to be deleted after calling this function!!!
        '''
        
        self.txtenc_model = None
        self.imgenc_model = None
        self.crossenc_model = None
        
        self.cache_imgenc.clear()
        self.cache_imgenc.clear()
        
        return
    
    # .................................................................................................................
    
    def get_device_str(self) -> tuple[str, str]:
        
        ex_param = next(self.parameters())
        device_str = str(ex_param.device.type)
        dtype_str = str(ex_param.dtype).replace("torch", "").replace(".", "")
        
        return device_str, dtype_str
    
    # .................................................................................................................
    
    def set_device(self, device: str | torch.device, dtype: str | torch.dtype | None = None) -> tuple[str, str]:
        
        # Parse dtype strings
        if isinstance(dtype, str):
            dtype_lut = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            dtype = dtype_lut.get(dtype, torch.float32)
        
        try:
            # Can fail if trying to move to gpu on a machine without gpu-support
            self.to(device, dtype)
        except:
            self.to("cpu", torch.float32)
        
        return self.get_device_str()
    
    # .................................................................................................................
    
    def encode_text(self, text_prompts_list: list[str]) -> TextEncoding:
        
        text_prompt = self._make_text_prompt_from_str_list(text_prompts_list)
        
        is_in_cache, text_encoding = self.cache_txtenc.retrieve(text_prompt)
        if not is_in_cache:
            with torch.inference_mode():
                text_encoding = self.txtenc_model(text_prompt)
            self.cache_txtenc.store(text_encoding, text_prompt)
        
        return text_encoding
    
    # .................................................................................................................
    
    def encode_image(self, image_bgr, cache_key = None) -> ImageEncoding:
        
        is_in_cache, image_encoding = self.cache_imgenc.retrieve(cache_key)
        if not is_in_cache:
            scaled_wh = self.imgenc_model.get_image_sizing(image_bgr.shape, 640, allow_upscaling = True)
            input_image_tensor = self.imgenc_model.prepare_image_for_model(image_bgr, scaled_wh)
            
            with torch.inference_mode():
                image_encoding = self.imgenc_model(input_image_tensor)
            self.cache_imgenc.store(image_encoding, cache_key)
        
        return image_encoding
    
    # .................................................................................................................
    
    def encode_text_with_image(self, text_encoding: TextEncoding, image_encoding: ImageEncoding) -> RawPredictions:
        
        with torch.inference_mode():
            raw_preds = self.crossenc_model(text_encoding, image_encoding)
        
        return raw_preds
    
    # .................................................................................................................
    
    def detect(self, text_prompts_list: list[str], image_bgr,
               box_confidence = 0.35, text_confidence = 0.8, image_cache_key = None) -> DetectionResults:
        
        # Calculate raw predictions (e.g. 900 boxes predicted)
        text_encoding = self.encode_text(text_prompts_list)
        image_encoding = self.encode_image(image_bgr, image_cache_key)
        raw_preds = self.encode_text_with_image(text_encoding, image_encoding)
        
        # Filter out only the 'high quality' predictions
        det_results = self.filter_predictions(
            text_encoding.text_prompt,
            raw_preds.logits,
            raw_preds.boxes_xywh_norm,
            text_encoding.input_ids,
            box_confidence,
            text_confidence,
        )
        
        return det_results
    
    # .................................................................................................................
    
    def filter_predictions(self, text_prompt, raw_logits, raw_boxes_xywh_norm, text_token_ids,
                           box_confidence = 0.5, text_confidence = 0.8) -> DetectionResults:
        
        # Fail on batches, since result is confusing. Better to have user loop over batches
        has_batch_dim = (raw_logits.ndim == 3)
        batch_size = raw_logits.shape[0]
        if has_batch_dim and batch_size > 1:
            raise NotImplementedError(
                "Prediction filtering does not work on batches! Loop over batch items if needed"
            )
        
        # Convert raw logits to probabilities and make sure we're working on the cpu
        logits_01 = raw_logits.sigmoid().squeeze().cpu()
        boxes_xywh_norm = raw_boxes_xywh_norm.squeeze().cpu()

        # Filter out boxes below box confidence
        max_prob_per_token, _ = logits_01.max(dim = 1)
        box_filter_mask = (max_prob_per_token > box_confidence)
        good_logits = logits_01[box_filter_mask, :]
        good_boxes_xywh = boxes_xywh_norm[box_filter_mask, :]
        
        # Assign text label & score to each box
        label_thresh = box_confidence * min(text_confidence, 1.0)
        labels_list = [self.decode_text(logit, text_token_ids, label_thresh) for logit in good_logits]
        scores, _ = good_logits.max(dim=1)
        
        have_detections = len(good_boxes_xywh) > 0
        return DetectionResults(text_prompt, good_boxes_xywh, labels_list, scores, have_detections)
    
    # .................................................................................................................
    
    def decode_text(self, logit_01: Tensor, text_token_ids: Tensor, text_threshold = 0.25) -> str:
        
        '''
        Helper used to decode text associated with a given logit output by the model
        Requires the logit itself, in 0-to-1 format (i.e. passed through sigmoid) as
        well as the input text token ids from the tokenizer
        '''
        
        return self.txtenc_model.decode_text(logit_01, text_token_ids, text_threshold)
    
    # .................................................................................................................
    
    @staticmethod
    def _make_text_prompt_from_str_list(text_prompts_list: list[str]) -> str:
        return ". ".join(text_prompts_list).removesuffix(".")
    
    # .................................................................................................................


class DummyCache:
    
    def __init__(self):
        pass
    
    def retrieve(self, *args, **kwargs):
        is_in_cache = False
        encoding = None
        return is_in_cache, encoding

    def store(self, data_to_store, cache_key):
        return
    
    def clear(self):
        return

