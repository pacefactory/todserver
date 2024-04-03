# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
from torch import nn

from .encoder_model import TransformerEncoder
from .decoder_model import TransformerDecoder

from .utils import (
    MLP,
    ContrastiveEmbed,
    gen_encoder_output_proposals,
)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,                # 256 from config
        num_heads=8,                # 8 from config
        num_queries=900,            # 900 from config, orig 300
        num_encoder_layers=6,       # 6 from config
        num_decoder_layers=6,       # 6 from config
        d_feedforward=2048,         # 2048 from config
        # dropout=0.0,                # 0.0 from config
        # activation="relu",          # 'relu' from config
        # normalize_before=False,     # False from config
        # return_intermediate_dec=True,  # True from build function, orig False
        d_query=4,                  # 4 from config
        # num_patterns=0,             # 0 from config
        # for deformable encoder
        num_feature_levels=4,       # 4 from config, orig 1
        enc_n_points=4,             # 4 from config
        dec_n_points=4,             # 4 from config
        # two stage
        # two_stage_type="no",        # 'standard' from config
        # embed_init_tgt=False,       # True from config
        # for text
        # use_text_enhancer=False,    # True from config
        # use_fusion_layer=False,     # True from config
        use_checkpoint=False,               # True from config, orig False
        use_transformer_ckpt=False,         # True from config, orig False
        # use_text_cross_attention=True,     # True from config, orig False
        # text_dropout=0.0,           # 0.0 from config, orig 0.1
        # fusion_dropout=0.0,         # 0.0 from config, orig 0.1
        # fusion_droppath=0.1,        # 0.1 from config, orig 0.0
    ):
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.d_query = d_query
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_enc_points = enc_n_points
        self.num_dec_points = dec_n_points
        
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, d_feedforward,
                                          num_heads, num_queries, num_feature_levels, enc_n_points,
                                          use_checkpoint, use_transformer_ckpt)

        self.decoder = TransformerDecoder(num_decoder_layers, d_model, d_feedforward, d_query,
                                          num_heads, num_feature_levels, dec_n_points)

        # self.dec_layers = num_decoder_layers
        # self.num_queries = num_queries  # useful for single stage model only
        # self.num_patterns = num_patterns

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.tgt_embed = nn.Embedding(self.num_queries, d_model)

        # anchor selection at the output of encoder
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        # self.two_stage_wh_embedding = None

        self.enc_out_class_embed = ContrastiveEmbed()
        self.enc_out_bbox_embed = MLP(d_model, d_model, 4, 3)


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) # Always tensor([[1., 1.]]) if not masking images?
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, 
                text_features, text_token_mask, text_attention_mask, text_position_ids,
                refpoint_embed = None, tgt = None, attn_mask=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        
        # For clarity
        inv_text_token_mask = ~text_token_mask
        # encoded_text = text_dict["encoded_text"]
        # inv_text_token_mask = ~text_dict["text_token_mask"]
        # text_pos_ids = text_dict["position_ids"]
        # text_self_attention_masks = text_dict["text_self_attention_masks"]
        
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_features,
            text_attention_mask=inv_text_token_mask,
            position_ids=text_position_ids,
            text_self_attention_masks=text_attention_mask,
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

        output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, memory_text, inv_text_token_mask)
        topk_logits = enc_outputs_class_unselected.max(-1)[0]
        topk_proposals = torch.topk(topk_logits, self.num_queries, dim=1)[1]  # bs, nq

        # gather boxes
        enc_outputs_coord_unselected = \
            (self.enc_out_bbox_embed(output_memory) + output_proposals)  # (bs, \sum{hw}, 4) unsigmoid
        refpoint_embed_undetach = \
            torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
        refpoint_embed_ = refpoint_embed_undetach.detach()
        init_box_proposal = \
            torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  # sigmoid

        # gather tgt
        tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
        tgt_ = (self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1))  # nq, bs, d_model

        # if refpoint_embed is not None:
        #     refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
        #     tgt = torch.cat([tgt, tgt_], dim=1)
        # else:
        refpoint_embed, tgt = refpoint_embed_, tgt_
        
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=memory_text,
            text_attention_mask=inv_text_token_mask,
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        
        hs_enc = tgt_undetach.unsqueeze(0)
        ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        internal_dict = {"hs_enc": hs_enc, "ref_enc": ref_enc, "init_box_proposal": init_box_proposal}
        
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, memory_text, internal_dict
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions


