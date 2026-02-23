import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn, Tensor
from typing import Optional, List
from .network_swinir import RSTB
from .vqgan import VQGAN
from .fema_utils import ResBlock
from einops.layers.torch import Rearrange


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class Predictor(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                codebook_size=1024,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
            
        self.norm_out=nn.LayerNorm(embed_dim)
   
        self.idx_pred_layer = nn.Sequential(
            nn.Linear(embed_dim, codebook_size, bias=False))
    
    def forward(self, x,return_embeds=False):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x=self.norm_out(x)
        logits = self.idx_pred_layer(x)    
        
        return logits
    
class Swich(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch,
        out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, alpha=1):
     
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = alpha * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out
    
class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


@ARCH_REGISTRY.register()
class Critic(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                codebook_size=1024,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(2):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
            
        self.norm_out=nn.LayerNorm(embed_dim)
        self.tok_emb = nn.Embedding(codebook_size, embed_dim)
        self.idx_pred_layer =  nn.Sequential(nn.Linear(embed_dim, 1),Rearrange('... 1 -> ...'))
    
    def forward(self, x,h,w,return_embeds=False):
        x =self.tok_emb(x)
        
        for m in self.swin_blks:
            x = m(x, (h, w))
        x=self.norm_out(x)
        logits = self.idx_pred_layer(x)    
    
        return logits

@ARCH_REGISTRY.register()
class DehazeTokenNet(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_semantic_loss=False,
                 use_residual=True,
                 blk_depth=12,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[0]
        codebook_emb_num = codebook_params[1].astype(int)
        codebook_emb_dim = codebook_params[2].astype(int)
        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = 1
        self.use_residual = use_residual

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale))
        encode_depth = int(
            np.log2(gt_resolution // self.scale_factor // self.codebook_scale))
        self.vqgan = VQGAN(in_channel, codebook_params, gt_resolution,
                           LQ_stage, norm_type, act_type, use_quantize,
                           use_semantic_loss, use_residual)
        if LQ_stage:
            self.transformer = Predictor()
        self.LQ_stage = LQ_stage

        self.fuse_convs_dict = nn.ModuleDict()
         # fuse_convs_dict
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i
            in_ch=channel_query_dict[cur_res]
            self.fuse_convs_dict[str(cur_res)] = Fuse_sft_block(in_ch, in_ch)

    def forward(self,
                input,
                hq_feats,
                token_mask,
                alpha=1,
                code_only=True,
                detach_16=False,
                adapter_img=None):
        
        enc_feats = self.vqgan.multiscale_encoder(input.detach())
        enc_feats = enc_feats[::-1]

        x = enc_feats[0]
        b, c, h, w = x.shape

        feat_to_quant = self.vqgan.before_quant(x)
        masked_feats = token_mask * feat_to_quant + ~token_mask * hq_feats

        logits = self.transformer.forward(masked_feats, return_embeds=False)
        
        if self.LQ_stage and code_only:
            return logits, feat_to_quant
            
        out_tokens = logits.argmax(dim=2)
        
        z_quant = self.vqgan.quantize.get_codebook_entry(
            out_tokens.reshape(b, 1, h, w))
        
        after_quant_feat = self.vqgan.after_quant(z_quant)

        if detach_16:
            after_quant_feat = after_quant_feat.detach()  # for training stage III

        x = after_quant_feat

        if hasattr(self.vqgan, 'retinex_adapter'):

            target_img = adapter_img if adapter_img is not None else input
            
            guidance = self.vqgan.retinex_adapter(target_img)

            if guidance.shape[-2:] != x.shape[-2:]:
                guidance = F.interpolate(guidance, size=x.shape[-2:], mode='bilinear')
            
            x = x + guidance

        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i
            if alpha > 0:
                x = self.fuse_convs_dict[str(cur_res)](enc_feats[i].detach(), x, alpha)

            x = self.vqgan.decoder_group[i](x)
            
        out_img = self.vqgan.out_conv(x)

        return logits, feat_to_quant, out_img
    
    def decode_indices(self, indices, hazy_img=None):

        assert len(
            indices.shape
        ) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.vqgan.quantize.get_codebook_entry(indices)
        x = self.vqgan.after_quant(z_quant)

        # ==========================================
        # [v1.0 Modified] PISA Adapter Injection (Inference)
        # ==========================================
        if hazy_img is not None and hasattr(self.vqgan, 'retinex_adapter'):
            guidance = self.vqgan.retinex_adapter(hazy_img)
            
            if guidance.shape[-2:] != x.shape[-2:]:
                guidance = F.interpolate(guidance, size=x.shape[-2:], mode='bilinear')
            
            x = x + guidance
        # ==========================================

        for m in self.vqgan.decoder_group:
            x = m(x)
        out_img = self.vqgan.out_conv(x)
        return out_img