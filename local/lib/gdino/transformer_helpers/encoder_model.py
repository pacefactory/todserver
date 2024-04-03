




# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn

from .utils import get_sine_pos_embed


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class TransformerEncoderLayer(nn.Module):
    
    # .................................................................................................................
    
    def __init__(self, d_model, num_heads, d_feedforward=2048, dropout=0.1):
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # self.ff_out = nn.Sequential(
        #         nn.Linear(d_model, d_feedforward),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(d_feedforward, d_model),
        #         nn.Dropout(dropout),
        #     )

    # .................................................................................................................
    
    def forward(self, src, src_mask, pos = None):
        
        # repeat attn mask
        if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            # bs, num_q, num_k
            src_mask = src_mask.repeat(self.num_heads, 1, 1)

        # run self-attention
        q = k = add_position_embedding(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # run feedforward on self-attention result
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        return self.norm2(src)


# =====================================================================================================================


class DeformableTransformerEncoderLayer(nn.Module):
    
    # .................................................................................................................
    
    def __init__(self, d_model=256, d_ffn=1024, n_heads=8, n_levels=4, n_points=4, dropout=0.1):
        
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    # .................................................................................................................
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    # .................................................................................................................
    
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        
        # self attention
        src2 = self.self_attn(
            query=add_position_embedding(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


# =====================================================================================================================


class TransformerEncoder(nn.Module):
    
    # .................................................................................................................
    
    def __init__(self, num_layers, d_model = 256, d_feedforward = 2048,
                 num_heads = 8, num_queries = 300, num_feature_levels = 4, num_enc_points = 4,
                 use_checkpoint = False, use_transformer_ckpt = False):
        
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.

        """
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

        # prepare layers
        self.layers = self.make_encoder_layers(num_layers, d_model, d_feedforward, num_heads, num_feature_levels, num_enc_points)
        self.text_layers = self.make_text_layers(num_layers, d_model, d_feedforward, num_heads)
        self.fusion_layers = self.make_fusion_layers(num_layers, d_model, d_feedforward, num_heads)

    # .................................................................................................................
    
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing = "ij"
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    # .................................................................................................................
    
    def forward(self, src, pos, spatial_shapes, level_start_index, valid_ratios,
                key_padding_mask, memory_text = None, text_attention_mask = None,
                pos_text = None, text_self_attention_masks = None, position_ids = None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        # generate pos_text
        # bs, n_text, text_dim = memory_text.shape
        # if pos_text is None and position_ids is None:
        #     pos_text = torch.arange(n_text, device=memory_text.device).float().unsqueeze(0).unsqueeze(-1).repeat(bs,1,1)
        #     pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
        # if position_ids is not None:
        pos_text = get_sine_pos_embed(position_ids[..., None], num_pos_feats=256, exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            
            if self.use_checkpoint:
                output, memory_text = checkpoint.checkpoint(self.fusion_layers[layer_id], output, memory_text,
                                                            key_padding_mask, text_attention_mask)
            else:
                output, memory_text = self.fusion_layers[layer_id](v=output, l=memory_text,
                                                                   attention_mask_v=key_padding_mask,
                                                                   attention_mask_l=text_attention_mask)

            memory_text = self.text_layers[layer_id](
                src=memory_text.transpose(0, 1),
                src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                pos=pos_text.transpose(0, 1) # if pos_text is not None else None),
            ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text
    
    # .................................................................................................................
    
    @staticmethod
    def make_encoder_layers(num_layers, d_model, d_feedforward, num_heads, num_feature_levels, num_points,
                            dropout = 0.0):
        
        encoder_layers_list = []
        for _ in range(num_layers):
            enc_layer = DeformableTransformerEncoderLayer(d_model, d_feedforward, num_heads,
                                                          num_feature_levels, num_points, dropout)
            encoder_layers_list.append(enc_layer)
            
        return nn.ModuleList(encoder_layers_list)
    
    # .................................................................................................................
    
    @staticmethod
    def make_text_layers(num_layers, d_model, d_feedforward, num_heads, dropout = 0.0):
        
        layer_num_heads = num_heads // 2
        layer_num_ff = d_feedforward // 2
        text_enhance_layers_list = []
        for _ in range(num_layers):
            text_layer = TransformerEncoderLayer(d_model, layer_num_heads, layer_num_ff, dropout)
            text_enhance_layers_list.append(text_layer)

        return nn.ModuleList(text_enhance_layers_list)
    
    # .................................................................................................................
    
    @staticmethod
    def make_fusion_layers(num_layers, d_model, d_feedforward, num_heads, dropout = 0.0, drop_path = 0.1):
        
        fusion_num_heads = num_heads // 2
        fusion_num_ff = d_feedforward // 2
        fusion_feature_layers_list = []
        for _ in range(num_layers):
            fusion_layer = BiAttentionBlock(v_dim=d_model, l_dim=d_model, embed_dim=fusion_num_ff,
                                            num_heads=fusion_num_heads, dropout=dropout, drop_path=drop_path)
            fusion_feature_layers_list.append(fusion_layer)
        
        return nn.ModuleList(fusion_feature_layers_list)


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def add_position_embedding(tensor, pos):
    return tensor if pos is None else tensor + pos

# .....................................................................................................................
