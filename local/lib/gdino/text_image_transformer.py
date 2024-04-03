#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------
# Modified/simplified from Grounding DINO:
# url: https://github.com/IDEA-Research/GroundingDINO
# ------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
from torch import nn

from .transformer_helpers.utils import MLP, ContrastiveEmbed
from .transformer_helpers.transformers import Transformer

from .model_datatypes import TextEncoding, ImageEncoding, RawPredictions, DetectionResults


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes


class TextAndImageTransformer(nn.Module):
    
    '''
    This is a simplified version of the text + image transformer from the Grounding DINO model
    See the .forward(...) method for details about what is actually happening
    '''
    
    # .................................................................................................................
    
    def __init__(self, hidden_dim = 256):
        
        # Inherit from parent
        super().__init__()
        
        
        self.transformer = Transformer(hidden_dim)
        
        # bbox embed
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        box_embed_layerlist = [_bbox_embed for i in range(self.transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        
        # class embded
        self.class_embed = nn.ModuleList([ContrastiveEmbed() for _ in range(self.transformer.num_decoder_layers)])
        
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed
        
        # self.transformer.enc_out_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # Separate copy of bbox embed (due to config)
        # self.transformer.enc_out_class_embed = ContrastiveEmbed()               # Separate copy of class embed (due to config)
        # self.refpoint_embed = None
    
    # .................................................................................................................
    
    # def forward(self, image_features_list, image_masks_list, image_positional_encodings_list, text_dict):
    def forward(self,  text_encoding: TextEncoding, image_encoding: ImageEncoding) -> RawPredictions:
        
        # Move all incoming data to the correct device (i.e. cpu vs gpu)
        sample_param = next(self.parameters())
        device, dtype = sample_param.device, sample_param.dtype
        text_encoding.to(device, dtype)
        image_encoding.to(device, dtype)
        # image_features = [item.to(device) for item in image_features_list]
        # image_masks = [item.to(device) for item in image_masks_list]
        # image_positional_encodings = [item.to(device) for item in image_positional_encodings_list]
        # new_text_dict = {k: value.to(device) for k, value in text_dict.items()}
        
        hs, reference, memory_text, _ = self.transformer(
            image_encoding.features_list,
            image_encoding.masks_list,
            image_encoding.posencs_list,
            text_encoding.features,
            text_encoding.token_mask,
            text_encoding.attn_mask,
            text_encoding.position_ids,
        )
        # hs, reference, memory_text, _ = \
        #     self.transformer(image_features, image_masks, image_positional_encodings, new_text_dict)
        
        # deformable-detr-like anchor update
        # outputs_coord_list = []
        # for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], self.bbox_embed, hs):
        #     layer_delta_unsig = layer_bbox_embed(layer_hs)
        #     layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
        #     layer_outputs_unsig = layer_outputs_unsig.sigmoid()
        #     outputs_coord_list.append(layer_outputs_unsig)
        # outputs_coord_list = torch.stack(outputs_coord_list)
        
        # bundle outputs
        # encoded_text = text_dict["encoded_text"]
        # inv_text_token_mask = ~text_dict["text_token_mask"]
        # outputs_class_list = []
        # for layer_cls_embed, layer_hs in zip(self.class_embed, hs):
        #     new_logit = layer_cls_embed(layer_hs, encoded_text, inv_text_token_mask)
        #     outputs_class_list.append(new_logit)
        # outputs_class = torch.stack(outputs_class_list)
        
        inv_text_token_mask = ~text_encoding.token_mask #~new_text_dict["text_token_mask"]
        last_h = hs[-1]
        
        last_ref_sig = reference[-2]
        last_bbox_embed = self.bbox_embed[-1]
        last_delta_unsig = last_bbox_embed(last_h)
        last_outputs_unsig = last_delta_unsig + inverse_sigmoid(last_ref_sig)
        pred_boxes = last_outputs_unsig.sigmoid()
        
        last_cls_embed = self.class_embed[-1]
        pred_logits = last_cls_embed(last_h, memory_text, inv_text_token_mask)
        
        return RawPredictions(pred_logits, pred_boxes)
        
        # outputs_class = torch.stack(
        #     [layer_cls_embed(layer_hs, text_dict) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)]
        # )
        # out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
                
        # return pred_logits, pred_boxes
    
    # .................................................................................................................
    
    # @staticmethod
    # def filter_batch_predictions(logits_batch, boxes_batch, input_token_ids, box_threshold = 0.3, text_threshold = 0.8) -> DetectionResults:
        
    #     # Convert raw logits to probabilities and make sure we're working on the cpu
    #     logits_prob_batch = logits_batch.sigmoid().cpu()
    #     boxes_batch = boxes_batch.cpu()
        
    #     # For each batch of data, filter out the 'best' predictions & find the corresponding text token ids
    #     good_boxes_batch = []
    #     box_token_ids_batch = []
    #     box_prob_batch = []
    #     for logits_prob, boxes in zip(logits_prob_batch, boxes_batch):
            
    #         # Grab only boxes with a 'high confidence' text token
    #         max_token_prob_per_detection, _ = logits_prob.max(dim=1)
    #         print("PROB SHAPE", max_token_prob_per_detection.shape)
    #         box_filter_mask = (max_token_prob_per_detection > box_threshold)
    #         print("BOXMASK SHAPE", box_filter_mask.shape)
    #         good_logits = logits_prob[box_filter_mask]
    #         good_boxes = boxes[box_filter_mask]
    #         print("GOOD LOG/BOX SHAPE", good_logits.shape, good_boxes.shape)
            
    #         # Extract the highest prob. token(s) associated with each box
    #         # -> in case of multiple labels (e.g. 'person' and 'vehicle') this step selects the appropriate label per box
    #         logit_threshold = min(text_threshold * box_threshold, box_threshold)
    #         box_token_ids = []
    #         box_probs = []
    #         for each_logit in good_logits:
    #             text_filter_mask = (each_logit > logit_threshold)
    #             non_zero_idx = text_filter_mask.nonzero(as_tuple=True)[0].tolist()
    #             good_token_ids = [input_token_ids[i] for i in non_zero_idx]
    #             box_token_ids.append(good_token_ids)
    #             box_probs.append(each_logit.max().item())
            
    #         good_boxes_batch.append(good_boxes)
    #         box_token_ids_batch.append(box_token_ids)
    #         box_prob_batch.append(box_probs)
        
    #     # Special handling for common case, where there is only 1 batch item, remove batch dim from output
    #     not_batched = (len(good_boxes_batch) == 1)
    #     if not_batched:
    #         good_boxes_batch = good_boxes_batch[0]
    #         box_token_ids_batch = box_token_ids_batch[0]
    #         box_prob_batch = box_prob_batch[0]
        
    #     return good_boxes_batch, box_token_ids_batch, box_prob_batch
    
    # # .................................................................................................................
    
    # @staticmethod
    # def filter_predictions(raw_logits, raw_boxes_xywh_norm, input_token_ids,
    #                        box_confidence = 0.5, text_confidence = 0.8):
        
    #     # Bail if we get batched data
    #     is_batched = len(raw_logits.shape) == 3
    #     if is_batched:
    #         num_batches = raw_logits.shape[0]
    #         assert num_batches == 1, "Error! Cannot filter batched input! Handle batch inputs in a loop if needed..."
        
    #     # Convert raw logits to probabilities and make sure we're working on the cpu
    #     logits = raw_logits.sigmoid().squeeze().cpu()
    #     boxes_xywh_norm = raw_boxes_xywh_norm.squeeze().cpu()
        
    #     # Generate mask used to pick out only the 'good' box detections
    #     # -> This is done by keeping any box entry whose corresponding logit
    #     #    has at least one feature value (of the 256 values) greater than some threshold
    #     max_prob_per_token, _ = logits.max(dim = 1)
    #     box_filter_mask = (max_prob_per_token > box_confidence)
        
    #     # Keep only the high confidence boxes & logits for further (text processing)
    #     good_logits_tensor = logits[box_filter_mask, :]
    #     good_boxes_tensor = boxes_xywh_norm[box_filter_mask, :]
        
    #     logit_mask_threshold = box_confidence * min(text_confidence, 1.0)
    #     box_token_ids, det_scores_list = [], []
    #     for each_logit in good_logits_tensor:
    #         text_filter_mask = (each_logit > logit_mask_threshold)
    #         non_zero_idx = text_filter_mask.nonzero(as_tuple=True)[0].tolist()
            
    #         good_token_ids = [input_token_ids[i] for i in non_zero_idx]
    #         score = each_logit.max().item()
            
    #         box_token_ids.append(good_token_ids)
    #         det_scores_list.append(score)
        
    #     return DetectionResults(good_boxes_tensor, box_token_ids, det_scores_list)
    
    # .................................................................................................................
    

# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

# .................................................................................................................
