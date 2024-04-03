# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
# Taken from:
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
# -> mmdet/models/backbones/swin_transformer.py
# *** Minor edits have been made (removing unused functions)
# --------------------------------------------------------


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        window_size_hw = (window_size, window_size)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size_hw, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        # x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 # drop=0.,
                 # attn_drop=0.,
                 # drop_path=0.,
                 include_patch_merging=False,
                 use_checkpoint=False):
        
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                )
                # drop=drop,
                # attn_drop=attn_drop,
                # drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        # patch merging layer
        self.include_patch_merging = include_patch_merging
        self.downsample = PatchMerging(dim) if include_patch_merging else None
    
    # .................................................................................................................

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device, dtype=x.dtype)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = checkpoint.checkpoint(blk, x, attn_mask) if self.use_checkpoint else blk(x, attn_mask)
        
        if self.include_patch_merging:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        
        return x, H, W, x, H, W


# =====================================================================================================================


class PatchEmbed(nn.Module):
    
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    
    # .................................................................................................................

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        
        # Inherit from parent class
        super().__init__()

        # Store config        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Set up output processing layers
        patch_tuple = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_tuple, stride = patch_tuple)
        self.norm = nn.LayerNorm(embed_dim)

    # .................................................................................................................
    
    def forward(self, images_bchw_tensor):
        
        '''
        Takes an input batch of images of shape: [batch_size, channels, height, width]
        and encodes patches of the image into N-dimensional vectors (where N = self.embed_dim)
        also applies a layernorm to the result. Encoding is done with a convolutional layer.
        Performs padding of input, if needed, to get an integer number of patches
        
        Result is of shape: [batch_size, embed_dim, height / patch_size, width / patch_size]
        '''
        
        # Get sizing for convenience
        _, _, h_in, w_in = images_bchw_tensor.size()
        
        # Figure out whether we need to pad width/height to create whole patches
        w_pad_remainder = (w_in % self.patch_size)
        h_pad_remainder = (h_in % self.patch_size)
        needs_h_padding = (h_pad_remainder != 0)
        needs_w_padding = (w_pad_remainder != 0)
        
        # Pad input if needed
        if needs_w_padding or needs_h_padding:
            pad_w_left = 0
            pad_w_right = self.patch_size - w_pad_remainder
            pad_h_top = 0
            pad_h_bot = self.patch_size - h_pad_remainder
            images_bchw_tensor = F.pad(images_bchw_tensor, (pad_w_left, pad_w_right, pad_h_top, pad_h_bot))
        
        # Create patch embeddings from input image, result has shape: [batch_size, embed_dim, h_patches, w_patches]
        # where w/h_patches is the number of patches in width/height, e.g. w_patches = (img_w / patch_size)
        patches_behw_tensor = self.proj(images_bchw_tensor)
        
        # Get output size, so we can restore tensor after flattening for norm layer
        _, embed_size, h_patches, w_patches = patches_behw_tensor.size()
        
        # Flatten width/height axes, normalize them, then restore original shape
        # -> The intermediate/flattened shape is [batch_size, w*h, embed_dim] after transposing
        patches_bNe_tensor = patches_behw_tensor.flatten(2).transpose(1, 2)
        patches_bNe_tensor = self.norm(patches_bNe_tensor)
        patches_behw_tensor = patches_bNe_tensor.transpose(1, 2).view(-1, embed_size, h_patches, w_patches)
        
        return patches_behw_tensor


# =====================================================================================================================


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, use_tiny_swin = True):
    # def __init__(self,
    #              use_tiny_swin = True,
    #              patch_size=4,
    #              in_chans=3,
    #              mlp_ratio=4.,                  # ?
    #              qkv_bias=True,                 # ?
    #              qk_scale=None,                 # ?
    #              drop_rate=0.,                  # ?
    #              attn_drop_rate=0.,             # ?
    #              drop_path_rate=0.2,            # ?
    #              out_indices=(1, 2, 3),         # Originally (0, 1, 2, 3), GDINO uses (1,2,3)
    #              use_checkpoint = False          # Originally set to False, GDINO uses True
    #              ): 
        
        # Inherit from parent class
        super().__init__()
        
        # Swap configs based on swin sizing
        pick_tiny_or_big = lambda tiny_val, big_val: tiny_val if use_tiny_swin else big_val
        pretrain_img_size = pick_tiny_or_big(224, 384)
        embed_dim = pick_tiny_or_big(96, 128)
        depth_per_stage = pick_tiny_or_big([2,2,6,2], [2,2,18,2])
        num_heads_per_stage = pick_tiny_or_big([3,6,12,24], [4,8,16,32])
        window_size = pick_tiny_or_big(7, 12)
        num_stages = len(depth_per_stage)
        num_features_per_stage = [int(embed_dim * (2 ** stage_idx)) for stage_idx in range(num_stages)]

        # Store config settings
        self.use_tiny_swin = use_tiny_swin
        self.pretrain_img_size = pretrain_img_size
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.out_indices_set = set((1,2,3))
        # self.use_checkpoint = use_checkpoint
        self.num_features_per_stage = num_features_per_stage
        
        # Set up model used to split images into non-overlapping patches
        self.patch_embed = PatchEmbed(embed_dim = embed_dim)
        # self.pos_drop = nn.Dropout(p=drop_rate)

        # Set up main transformer stages
        self.layers = self.create_transformer_stages(
            depth_per_stage,
            num_features_per_stage,
            num_heads_per_stage,
            window_size)
        #, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, use_checkpoint)


        # Add a norm layer for each output
        # -> Output of model is actually made of feature vectors taken at intermediate stages of the model
        # -> These intermediate layer outputs are passed through normalization layers to build the output
        self.norm = nn.ModuleList(
            nn.LayerNorm(num_features_per_stage[stage_idx]) for stage_idx in self.out_indices_set
        )
        # self.out_norm_layers_dict = nn.ModuleDict()
        # for stage_idx in self.out_indices_set:
        #     out_norm_layer = nn.LayerNorm(num_features_per_stage[stage_idx])
        #     self.out_norm_layers_dict[str(stage_idx)] = out_norm_layer
        #     # layer_name = f'norm{layer_idx}'
        #     # self.add_module(layer_name, layer)
        pass

    # .................................................................................................................
    
    def forward(self, x, image_masks_tensor):
        
        # Convert image patches into 'rows of tokens' shape
        x = self.patch_embed(x)
        _, _, h_patches, w_patches = x.size()
        x = x.flatten(2).transpose(1, 2)
        
        # x = self.pos_drop(x)
        # for layer_idx in range(self.num_layers):
            # layer = self.layers[layer_idx]
        
        # Run input through each of the transformer blocks, picking off intermediate results for outputs
        stage_features = []
        for stage_idx, layer in enumerate(self.layers):
            
            # Run each block layer
            x_out, H, W, x, h_patches, w_patches = layer(x, h_patches, w_patches)
            
            # Save result, if we're on one of the target output stages
            is_output_layer = (stage_idx in self.out_indices_set)
            if is_output_layer:
                
                # Apply normalization to intermediate layer result
                norm_idx = stage_idx - 1
                x_out = self.norm[norm_idx](x_out)
                # out_norm_layer = self.out_norm_layers_dict[str(stage_idx)]
                # x_out = out_norm_layer(x_out)

                # Reshape layer result to be of shape: [???]
                num_features = self.num_features_per_stage[stage_idx]
                x_out = x_out.view(-1, H, W, num_features).permute(0, 3, 1, 2).contiguous()
                stage_features.append(x_out)

        # # Create masks for each output
        # masks_list = []
        # # outs_dict = {}
        # for idx, out_i in enumerate(stage_features):
        #     mask = F.interpolate(image_masks_tensor[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
        #     #outs_dict[idx] = NestedTensor(out_i, mask)
        #     masks_list.append(mask)

        return stage_features
    
    # .................................................................................................................
    
    @staticmethod
    def create_transformer_stages(depth_per_stage, num_features_per_stage, num_heads_per_stage, window_size): 
                                  # window_size, mlp_ratio, qkv_bias, qk_scale,
                                  # drop_rate, attn_drop_rate, drop_path_rate, use_checkpoint):
        
        
        # Number of repeated transformer stages (i.e. BasicLayer)
        num_stages = len(depth_per_stage)
        
        # stochastic depth
        # num_total_transformer_blocks = sum(depth_per_stage)
        # dpr = torch.linspace(0, drop_path_rate, num_total_transformer_blocks).tolist()  # stochastic depth decay rule
        
        # Build stages
        transformer_stages = nn.ModuleList()
        for stage_idx in range(num_stages):
            
            # For clarity, set up stage config
            stage_depth = depth_per_stage[stage_idx]
            stage_heads = num_heads_per_stage[stage_idx]
            stage_dim = num_features_per_stage[stage_idx]
            is_not_last_layer = (stage_idx < (num_stages - 1))
            
            # Figure out dpr per layer
            # stage_block_start_idx = sum(depth_per_stage[:stage_idx])
            # stage_block_end_idx = sum(depth_per_stage[:(stage_idx + 1)])
            # stage_dpr = dpr[stage_block_start_idx:stage_block_end_idx]
            
            # Make new layer and add to listing
            new_layer = BasicLayer(
                dim=stage_dim,
                depth=stage_depth,
                num_heads=stage_heads,
                window_size=window_size,
                # drop_path=stage_dpr,
                include_patch_merging=is_not_last_layer,
                )
            transformer_stages.append(new_layer)
        
        return transformer_stages
    
    # .................................................................................................................
    
    @staticmethod
    def image_uint8_to_tensor(image_rgb_uint8):
        
        '''
        Function which converts an image from standard uint8 format into
        a properly normalized tensor for input into the swin transformer model.
        Uses rgb mean: (0.485, 0.456, 0.406)
        Uses rgb std: (0.229, 0.224, 0.225)
        
        Inputs:
            image_rgb_uint8 -> Image with values ranging from 0 to 255, in RGB order
                               Shape is expected to be [height, width, channels]
        
        Returns:
            image_tensor -> A float32 tensor, with shape [batch, channels, height, width]
                            Values are also normalized (mean & standard deviation)
        '''        
        
        # Apply normalization
        image_tensor = torch.tensor(image_rgb_uint8, dtype = torch.float32) * (1.0 / 255.0)
        mean_tensor = torch.tensor((0.485, 0.456, 0.406))
        std_tensor = 1.0 / torch.tensor((0.229, 0.224, 0.225))
        image_tensor = (image_tensor - mean_tensor) * std_tensor
        
        # Re-order axes so that we go from: [h, w, ch] to [c, h, w] and add batch -> [b, c, h, w]
        return image_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # .................................................................................................................

