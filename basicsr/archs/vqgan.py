import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY

from .network_swinir import RSTB
from .fema_utils import ResBlock, CombineQuantBlock
from .vgg_arch import VGGFeatureExtractor
from typing import Any, Dict, Tuple

# ==========================================
# [v1.0 New Feature] Physics-Informed Structural Adapter (PISA)
# Based on AAAI 2026 ClearNight / ControlNet Logic
# ==========================================
class LearnableRetinexAdapter(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        
        # 1. Learnable Decom-Net (Full Resolution)
        # Dynamically decomposes Hazy Image into Reflectance (R) and Illumination (L)
        self.decom_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 4, 3, 1, 1) # Output: 3 channels (R) + 1 channel (L)
        )
        
        # 2. Dual-Prior Encoder (Downsampling Stream)
        # Compresses 256x256 structure into 16x16 latent features (16x downsample)
        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(4, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            # 128 -> 64
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            # 64 -> 32
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.SiLU(),
            # 32 -> 16
            nn.Conv2d(256, out_channels, 3, stride=2, padding=1),
            nn.SiLU()
        )
        
        # 3. Zero-Convolution (ControlNet Core)
        # Initializes with 0 to ensure stability at the start of training
        self.zero_conv = nn.Conv2d(out_channels, out_channels, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, img):
        # Step A: Physics Decomposition
        # We don't enforce strict physical constraints (like sigmoid) to allow 
        # the network to learn "optimal" features for dehazing
        decom_feat = self.decom_net(img)
        
        # Step B: Structural Encoding
        # Extracts edges/textures from R and density cues from L
        guidance = self.encoder(decom_feat)
        
        # Step C: Injection
        return self.zero_conv(guidance)


class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE Quantizer
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
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)
        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)
        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        codebook = self.embedding.weight
        d = self.dist(z_flattened, codebook)

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)
            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)
            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            codebook_loss = self.beta * ((z_q_gt.detach() - z) ** 2).mean()
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape
        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q

class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256,
                blk_depth=6, num_heads=8, window_size=8, **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x

class MultiScaleEncoder(nn.Module):
    def __init__(self, in_channel, max_depth, input_res=256,
                 channel_query_dict=None, norm_type='gn', act_type='leakyrelu',
                 LQ_stage=True, **swin_opts):
        super().__init__()
        ksz = 3
        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)
        self.blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2
        self.LQ_stage = LQ_stage

    def forward(self, input):
        outputs = []
        x = self.in_conv(input)
        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)
        return outputs

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()
        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]
        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)

@ARCH_REGISTRY.register()
class VQGAN(nn.Module):
    def __init__(self,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_semantic_loss=False,
                 use_residual=True,
                 ):
        super().__init__()

        codebook_params = np.array(codebook_params)
        self.codebook_scale = codebook_params[0]
        codebook_emb_num = codebook_params[1].astype(int)
        codebook_emb_dim = codebook_params[2].astype(int)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.use_residual = use_residual

        channel_query_dict = {
            8: 256, 16: 256, 32: 256, 64: 256,
            128: 128, 256: 64, 512: 32,
        }

        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale))
        encode_depth = int(np.log2(gt_resolution // self.codebook_scale))
        self.multiscale_encoder = MultiScaleEncoder(in_channel, encode_depth,
                                                    self.gt_res,
                                                    channel_query_dict,
                                                    norm_type, act_type,
                                                    LQ_stage)

        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(
                DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        self.quantize = VectorQuantizer(
            codebook_emb_num,
            codebook_emb_dim,
            LQ_stage=self.LQ_stage,
        )
        quant_conv_in_ch = channel_query_dict[self.codebook_scale]
        self.before_quant = nn.Conv2d(quant_conv_in_ch, codebook_emb_dim, 1)
        self.after_quant = CombineQuantBlock(codebook_emb_dim, 0,
                                                   quant_conv_in_ch)
        
        # ==========================================
        # [v1.0] Initialize PISA Adapter
        # Note: We match the dimension of 'quant_conv_in_ch' which is input to decoder
        # ==========================================
        self.retinex_adapter = LearnableRetinexAdapter(in_channels=3, out_channels=quant_conv_in_ch)


        # semantic loss
        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss:
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1, 0),
                nn.ReLU(),
            )
            self.vgg_feat_layer = 'relu4_4'
            self.vgg_feat_extractor = VGGFeatureExtractor(
                [self.vgg_feat_layer])

    def encode(self, image, gt_indices=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        enc_feats = self.multiscale_encoder(image.detach())[-1]
        feat_to_quant = self.before_quant(enc_feats)
        z_quant, codebook_loss, indices = self.quantize(feat_to_quant, gt_indices)
        return z_quant, codebook_loss, indices
    
    def encode_feats(self, image):
        enc_feats = self.multiscale_encoder(image.detach())[-1]
        feat_to_quant = self.before_quant(enc_feats)
        return feat_to_quant

    def encode_and_decode(self, input, gt_indices=None, current_iter=None, hazy_img=None):
        """
        Modified to support hazy_img injection during training if needed.
        """
        enc_feats = self.multiscale_encoder(input.detach())
        enc_feats = enc_feats[::-1]

        if self.use_semantic_loss:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(input)[self.vgg_feat_layer]

        codebook_loss_list = []
        semantic_loss_list = []
        x = enc_feats[0]
        feat_to_quant = self.before_quant(x)

        if gt_indices is not None:
            z_quant, codebook_loss, indices = self.quantize(feat_to_quant, gt_indices)
        else:
            z_quant, codebook_loss, indices = self.quantize(feat_to_quant)

        if self.use_semantic_loss:
            semantic_z_quant = self.conv_semantic(z_quant)
            semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
            semantic_loss_list.append(semantic_loss)

        if not self.use_quantize:
            z_quant = feat_to_quant

        after_quant_feat = self.after_quant(z_quant)
        x = after_quant_feat

        # ==========================================
        # [v1.0] Adapter Injection (Structure Guidance)
        # ==========================================
        if hazy_img is not None:
            guidance = self.retinex_adapter(hazy_img)
            # Resize if necessary (robustness)
            if guidance.shape[-2:] != x.shape[-2:]:
                guidance = F.interpolate(guidance, size=x.shape[-2:], mode='bilinear')
            x = x + guidance
        
        for m in self.decoder_group:
            x = m(x)
        
        codebook_loss_list.append(codebook_loss)
        out_img = self.out_conv(x)

        codebook_loss = sum(codebook_loss_list)
        semantic_loss = sum(semantic_loss_list) if len(
            semantic_loss_list) else codebook_loss * 0

        return out_img, codebook_loss, semantic_loss, indices

    def decode_indices(self, indices, hazy_img=None):
        """
        [v1.0] Modified to accept hazy_img for structural guidance
        """
        assert len(indices.shape) == 4, f'shape must be (b, 1, h, w), got {indices.shape}'

        z_quant = self.quantize.get_codebook_entry(indices)
        x = self.after_quant(z_quant)

        # ==========================================
        # [v1.0] Adapter Injection
        # ==========================================
        if hazy_img is not None:
            guidance = self.retinex_adapter(hazy_img)
            # Ensure spatial dimensions match (handling potential rounding issues in downsampling)
            if guidance.shape[-2:] != x.shape[-2:]:
                guidance = F.interpolate(guidance, size=x.shape[-2:], mode='bilinear')
            x = x + guidance

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img

    @torch.no_grad()
    def test(self, input):
        # ... (Original test logic preserved) ...
        # Note: If testing VQGAN standalone, we usually just reconstruction.
        # If testing IPC, we use decode_indices.
        
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False

        wsz = 8 // self.scale_factor * 8
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])],
                          2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])],
                          3)[:, :, :, :w_old + w_pad]

        dec, _, _, _ = self.encode_and_decode(input)

        output = dec
        output = output[..., :h_old * self.scale_factor, :w_old *
                        self.scale_factor]

        self.use_semantic_loss = org_use_semantic_loss
        return output

    def forward(self, input, gt_indices=None, hazy_img=None):
        # [v1.0] Added hazy_img to forward args
        if gt_indices is not None:
            dec, codebook_loss, semantic_loss, indices = self.encode_and_decode(
                input, gt_indices, hazy_img=hazy_img)
        else:
            dec, codebook_loss, semantic_loss, indices = self.encode_and_decode(
                input, hazy_img=hazy_img)

        return dec, codebook_loss, semantic_loss, indices