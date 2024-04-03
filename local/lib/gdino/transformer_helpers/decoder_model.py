




# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn

from .utils import MLP, gen_sineembed_for_position


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class DeformableTransformerDecoderLayer(nn.Module):
    
    def __init__(self, d_model=256, d_ffn=1024, n_heads=8, n_levels=4, n_points=4, dropout=0.1):
        
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt,  # nq, bs, d_model
        tgt_query_pos = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask = None,
        tgt_reference_points = None,  # nq, bs, 4
        memory_text = None,  # bs, num_token, d_model
        text_attention_mask = None,  # bs, num_token
        # for memory
        memory = None,  # hw, bs, d_model
        memory_key_padding_mask = None,
        memory_level_start_index = None,  # num_levels
        memory_spatial_shapes = None,  # bs, num_levels, 2
        memory_pos = None,  # pos for memory
        # sa
        self_attn_mask = None,  # mask used for self-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """

        # self attention
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.ca_text(
            self.with_pos_embed(tgt, tgt_query_pos),
            memory_text.transpose(0, 1),
            memory_text.transpose(0, 1),
            key_padding_mask=text_attention_mask,
        )[0]
        tgt = tgt + self.catext_dropout(tgt2)
        tgt = self.catext_norm(tgt)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


# =====================================================================================================================


class TransformerDecoder(nn.Module):
    
    # .................................................................................................................
    
    def __init__(self, num_layers, d_model, d_feedforward, d_query, num_heads, num_feature_levels, num_points):

        assert d_query in [2, 4], "d_query should be 2/4 but {}".format(d_query)
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.d_query = d_query
        self.num_heads = num_heads
        self.num_feature_levels = num_feature_levels
        self.num_points = num_points
       
        # Set up main layers
        self.layers = self.make_decoder_layers(num_layers, d_model, d_feedforward, num_heads, 
                                               num_feature_levels, num_points)
        
        self.norm = nn.LayerNorm(d_model)

        self.ref_point_head = MLP(d_query // 2 * d_model, d_model, d_model, 2)
        # self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        # self.class_embed = None


        # self.ref_anchor_head = None

    # .................................................................................................................
    
    def forward(
        self,
        tgt,
        memory,
        tgt_mask = None,
        tgt_key_padding_mask = None,
        memory_key_padding_mask = None,
        pos = None,
        refpoints_unsigmoid = None,  # num_queries, bs, 2
        # for memory
        level_start_index = None,  # num_levels
        spatial_shapes = None,  # bs, num_levels, 2
        valid_ratios = None,
        # for text
        memory_text = None,
        text_attention_mask = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = (reference_points[:, :, None]*torch.cat([valid_ratios, valid_ratios], -1)[None, :])  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            # main process
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # iter update
            if self.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.d_query] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.d_query].sigmoid()

                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [[itm_out.transpose(0, 1) for itm_out in intermediate],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]]
    
    # .................................................................................................................
    
    @staticmethod
    def make_decoder_layers(num_layers, d_model, d_feedforward, num_heads, num_feature_levels, num_points,
                            dropout = 0.0):
        
        decoder_layers_list = []
        for _ in range(num_layers):
            dec_layer = DeformableTransformerDecoderLayer(d_model, d_feedforward, num_heads, 
                                                          num_feature_levels, num_points, dropout)
            decoder_layers_list.append(dec_layer)
        
        return nn.ModuleList(decoder_layers_list)


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def add_position_embedding(self, tensor, pos):
    return tensor if pos is None else tensor + pos

# .....................................................................................................................

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

# .....................................................................................................................
