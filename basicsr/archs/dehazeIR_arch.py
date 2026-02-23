import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math
from typing import Optional, List
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn, Tensor
from .network_swinir import RSTB
from .vqgan import VQGAN
from .fema_utils import ResBlock, CombineQuantBlock
from basicsr.utils.mask_schedule import schedule
from typing import Callable
from einops.layers.torch import Rearrange
class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """
    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def dist(self, x, y):
        return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization. 
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0],
                                    codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0],
                                        codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            codebook_loss = self.beta * ((z_q_gt.detach() - z)**2).mean()
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(
            z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

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


class Transformer(nn.Module):
    def __init__(
        self,
        codebook_size=1024,
        num_layers=9,
        num_embeds=512,
        intermediate_size=512 * 2,
        num_heads=8,
    ) -> None:
        super().__init__()
 
        # transformer
        self.ft_layers = nn.Sequential(*[
            TransformerSALayer(embed_dim=num_embeds,
                               nhead=num_heads,
                               dim_mlp=intermediate_size,
                               dropout=0.0) for _ in range(num_layers)
        ])
        # self.tok_emb = nn.Embedding(codebook_size, num_embeds)
        
        # logits_predict head
        # self.idx_pred_layer = nn.Sequential(
        #     nn.LayerNorm(num_embeds),
        #     nn.Linear(num_embeds, codebook_size, bias=False))
        
        self.token_critic =  nn.Sequential(nn.Linear(num_embeds, 1),Rearrange('... 1 -> ...'))
        

    def forward(self, x ,token_critic=False):

        # quant_feat, codebook_loss, quant_stats = self.quantizer(lq_feat)
        # BCHW -> BC(HW) -> (HW)BC
        #TODO add a loss, use conv,there can be surpervised by the loss of (z_d and z)
        # print("x")
        # print(x.shape)
        
        
        x =x.flatten(2).permute(2, 0, 1)
        query_emb = x
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb)
        # output logits
            
        # logits = self.idx_pred_layer(query_emb)  # (hw)bn
        # logits = logits.permute(1, 0, 2)  # (hw)bn -> b(hw)n 
            
        if token_critic == False:
            return query_emb  # (hw)bn
        
        logits_mask=self.token_critic(query_emb)
        logits_mask = logits_mask.permute(1, 0)  # (hw)b -> b(hw)
    
        return query_emb.permute(1, 0, 2),logits_mask




@ARCH_REGISTRY.register()
class DehazeNet(nn.Module):
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
                           use_semantic_loss, use_residual, **ignore_kwargs)
        if LQ_stage:
            self.transformer = Transformer()

        self.LQ_stage = LQ_stage

        # block_args = dict(
        #             norm_type=norm_type,
        #             conv_fn=nn.Conv2d,
        #             activation_fn=self.activation_fn,
        #             use_conv_shortcut=False,
        #         )
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
                gt_indices=None,
                current_iter=None):
        enc_feats = self.vqgan.multiscale_encoder(input.detach())

        enc_feats = enc_feats[::-1]

        x = enc_feats[0]
        b, c, h, w = x.shape

        feat_to_quant = self.vqgan.before_quant(x)
        
        if hq_feats == None:
            hq_feats=feat_to_quant.detach()
        masked_feats = token_mask * hq_feats + ~token_mask * feat_to_quant
        
        z_quant= self.transformer.forward(masked_feats,token_critic=False)
        z_quant=z_quant.permute(1, 2, 0).reshape(b,-1,h,w)

        if self.LQ_stage and code_only:
            return z_quant
        ################# Quantization ###################
        # out_tokens = logits.argmax(dim=2)
        # z_quant = self.vqgan.quantize.get_codebook_entry(
        #     out_tokens.reshape(b,1,h,w))
        
        after_quant_feat = self.vqgan.after_quant(z_quant)

        if detach_16:
            after_quant_feat = after_quant_feat.detach()  # for training stage III
# 32 64 128

        x=after_quant_feat
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i
           
            # if self.LQ_stage and self.use_residual:
            #     x = x + enc_feats[i]
            # else:
            #     x = x
            if alpha>0:
                x = self.fuse_convs_dict[str(cur_res)](enc_feats[i].detach(), x, alpha)

            x = self.vqgan.decoder_group[i](x)
            
        out_img = self.vqgan.out_conv(x)

        return z_quant, feat_to_quant,out_img,
    
    def inference(self,
                input,
                alpha=1,
                code_only=True,
                detach_16=False,
                gt_indices=None,
                current_iter=None):
        enc_feats = self.vqgan.multiscale_encoder(input.detach())

        enc_feats = enc_feats[::-1]

        x = enc_feats[0]
        b, c, h, w = x.shape

        feat_to_quant = self.vqgan.before_quant(x)
        
        masked_target = feat_to_quant
     
        logits = self.transformer(masked_target)

        if self.LQ_stage and code_only:
            return logits, feat_to_quant
        ################# Quantization ###################
        out_tokens = logits.argmax(dim=2)
        z_quant = self.vqgan.quantize.get_codebook_entry(
            out_tokens.reshape(b,1,h,w))
        
        after_quant_feat = self.vqgan.after_quant(z_quant)    


        if detach_16:
            after_quant_feat = after_quant_feat.detach()  # for training stage III
# 32 64 128
        x=after_quant_feat
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i
           
            # if self.LQ_stage and self.use_residual:
            #     x = x + enc_feats[i]
            # else:
            # #     x = x
            if alpha>0:
                x = self.fuse_convs_dict[str(cur_res)](enc_feats[i].detach(), x, alpha)

            x = self.vqgan.decoder_group[i](x)
            
        out_img = self.vqgan.out_conv(x)

        return logits, feat_to_quant,out_img,  

    def decode_indices(self, indices):
        assert len(
            indices.shape
        ) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.vqgan.quantize.get_codebook_entry(indices)
        x = self.vqgan.after_quant(z_quant)

        for m in self.vqgan.decoder_group:
            x = m(x)
        out_img = self.vqgan.out_conv(x)
        return out_img

    @torch.no_grad()
    def test(self, input):
       
        # padding to multiple of window_size * 8
        wsz = 16
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])],
                          2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])],
                          3)[:, :, :, :w_old + w_pad]
        

        # out_img, codebook_loss, semantic_loss, indices=self.vqgan(input)
        # logits, feat_to_quant,= self.inference(input,t_map,alpha=0,code_only=True,detach_16=False)
        # output_token=logits.argmax(dim=2)
        # b,c,h,w=feat_to_quant.shape
        # out_img=self.decode_indices(output_token.reshape(1,1,h,w))
        logits, feat_to_quant,out_img,= self.inference(input,alpha=1,code_only=False,detach_16=False)

        output = out_img
        output = output[..., :h_old , :w_old ]
        return output

  