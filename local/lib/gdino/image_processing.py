#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Modified/simplified from Grounding DINO:
@ https://github.com/IDEA-Research/GroundingDINO
'''


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import torch
from torch import nn
import torch.nn.functional as F

from .image_helpers.swin_transformer import SwinTransformer
from .image_helpers.positional_encoding import PositionEmbeddingSine

from .model_datatypes import ImageEncoding


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class ImageEncoder(nn.Module):
    
    '''
    This is a simplified version of the image processing stages from the Grounded DINO model
    See the .forward(...) method for details about what is actually happening during image processing
    '''
    
    # .................................................................................................................
    
    def __init__(self, use_tiny_swin = True, hidden_dim = 256, posenc_temp = 20):
        
        # Inherit from parent class
        super().__init__()
        
        self.backbone = SwinTransformer(use_tiny_swin)
        self.input_proj = self.create_projection_layers(hidden_dim, self.backbone.num_features_per_stage)
        
        num_pos_steps = hidden_dim // 2
        self.posenc_model = PositionEmbeddingSine(num_pos_steps, posenc_temp, normalize = True)
    
    # .................................................................................................................
    
    def forward(self, input_images_tensor, image_masks_tensor = None) -> ImageEncoding:
        
        '''
        Inputs:
            input_images_tensor - Should have a shape of [batch_size, 3, height, width]
            image_masks_tensor - Should have a shape of [batch_size, height, width], each mask should have
                                 false values wherever a valid image pixel is, and true values for pixels that are
                                 outside of the original image (i.e. pixels added through padding). If
                                 this value is None, masks will be generated automatically (with all false values)
        '''
        
        # Make blank masks
        if image_masks_tensor is None:
            batch_size, num_channels, img_h, img_w = input_images_tensor.shape
            mask_shape = (batch_size, img_h, img_w)
            device = input_images_tensor.device
            image_masks_tensor = torch.zeros(mask_shape, dtype = torch.bool, device = device)
        
        # Encode image & generate a duplicated encoding of last layer (due to original GDINO structure)
        stage_outputs = self.backbone(input_images_tensor, image_masks_tensor)
        stage_outputs.append(stage_outputs[-1])
        
        out_features_list, out_masks_list, out_posencs_list = [], [], []
        for stage_idx, stage_features in enumerate(stage_outputs):
            
            out_features = self.input_proj[stage_idx](stage_features)
            
            # Scale mask to match feature sizing
            _, _, h_out, w_out = out_features.shape
            out_mask = F.interpolate(image_masks_tensor[None].float(), size=(h_out, w_out)).to(torch.bool)[0]
            
            # Generate positional encodings
            out_posenc = self.posenc_model(out_features, out_mask)
            
            # Store all results for output
            out_features_list.append(out_features)
            out_posencs_list.append(out_posenc)
            out_masks_list.append(out_mask)
        
        return ImageEncoding(out_features_list, out_posencs_list, out_masks_list)
    
    # .................................................................................................................
    
    def get_image_sizing(self, image_shape, max_side_length_px = 640, *, allow_upscaling = False):
        
        '''
        Find target image width & height for a given resizing target
        
        Inputs:
            image_shape -> results from nparray.shape or tensor.shape. Expected order is [height, width, channels]
            max_side_length_px -> maximum allowed side length (in pixels) of resized image
            allow_upscaling -> If true, the output size will be scaled to the max side length, even if
                               the original image was smaller
        
        Returns:
            resize_width_px, resize_height_px
        '''
        
        # Figure out image sizing
        img_h, img_w = image_shape[0:2]
        largest_side_px = max(img_h, img_w)
        
        # Bail if we have a small image and we're not using upscaling
        is_small_image = (largest_side_px < max_side_length_px)
        if is_small_image and not allow_upscaling:
            return (img_w, img_h)
        
        # Calculate scaled width/height so the largest side is set to the given maximum
        scale_factor = max_side_length_px / largest_side_px
        w_scaled = int(round(img_w*scale_factor))
        h_scaled = int(round(img_h*scale_factor))
        
        return w_scaled, h_scaled
    
    # .................................................................................................................
    
    def prepare_image_for_model(self, image_bgr, target_wh):
        
        ex_param = next(self.parameters())
        device = ex_param.device
        dtype = ex_param.dtype
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, dsize = target_wh)
        
        return self.backbone.image_uint8_to_tensor(image_rgb).to(device, dtype)
    
    # .................................................................................................................
    
    @staticmethod
    def create_projection_layers(proj_out_size, num_inputs_per_layer):
        
        '''
        Helper used to create output projection layers (called 'input_proj' from original GDINO model)
        These layers are used to map the 3 output tensors from the swin model + an additional
        duplicated copy of the 3rd swin tensor into 4 outputs with the same channel count
        for input into the final text-image-transformer model
        '''
        
        # Shared groupnorm config
        num_groups = 32
        
        # Create a projection layer for each of the final (last 3) output feature layers
        feature_proj_list = []
        for input_size in num_inputs_per_layer[1:]:
            proj_layer = nn.Sequential(
                nn.Conv2d(input_size, proj_out_size, kernel_size=1),
                nn.GroupNorm(num_groups, proj_out_size),
            )
            feature_proj_list.append(proj_layer)
        
        # Create additional projection layer, so that we have 4 total, to match original GDINO model
        num_inputs_final_layer = num_inputs_per_layer[-1]
        extra_proj_layer = nn.Sequential(
            nn.Conv2d(num_inputs_final_layer, proj_out_size, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups, proj_out_size),
        )
        feature_proj_list.append(extra_proj_layer)
        
        return nn.ModuleList(feature_proj_list)
    
    # .................................................................................................................
