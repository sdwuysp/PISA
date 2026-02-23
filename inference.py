from torch.nn import functional as F
import argparse
import cv2
import glob
import os
import torch
import numpy as np
from basicsr.archs.dehazeToken_arch import DehazeTokenNet
from basicsr.utils import tensor2img, imwrite, img2tensor
from basicsr.utils.post_process import adaptive_haze_correction
import pyiqa
import time 
import gc

def load_model(args, device):
    print(f"Loading model from: {args.model_path}")
    net_g = DehazeTokenNet(
        codebook_params=[64, 1024, 256],
        blk_depth=16,
        LQ_stage=True,
        predictor_name='swinLayer'
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    load_net = checkpoint['params_ema'] if 'params_ema' in checkpoint else checkpoint.get('params', checkpoint)
    
    if list(load_net.keys())[0].startswith('module.'):
        load_net = {k[7:]: v for k, v in load_net.items()}

    net_g.load_state_dict(load_net, strict=True)
    net_g.eval()
    return net_g

def pre_process_and_resize(img_path, device, max_size=1500, window_size=8):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: raise ValueError(f"can't read: {img_path}")
    
    h_orig, w_orig = img.shape[:2]
    
    scale = 1.0
    if max(h_orig, w_orig) > max_size:
        scale = max_size / max(h_orig, w_orig)
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print(f"   📉 Downsampling: {w_orig}x{h_orig} -> {new_w}x{new_h} (Scale: {scale:.3f})")
    
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)
    
    _, _, h, w = img.size()
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h != 0 or pad_w != 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        
    return img, h, w, h_orig, w_orig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='examples', help='Input folder')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--model_path', type=str, required=True, default='pretrained_models/predictor.pth', help='Model path')
    
    parser.add_argument('--n_iter', type=int, default=1, help='Iteration count')
    parser.add_argument('--alpha', type=float, default=0.85, help='SFT Alpha')
    
    parser.add_argument('--max_size', type=int, default=4000, help='Max image size for inference')
    
    parser.add_argument('--metrics', nargs='+', help='Metrics (psnr, ssim, etc.)')
    parser.add_argument('--gt', type=str, default=None, help='GT folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    metric_funcs = {}
    metric_scores = {}
    if args.metrics:
        print(f"Initializing metrics: {args.metrics}...")
        for m in args.metrics:
            try:
                metric_funcs[m] = pyiqa.create_metric(m.lower(), device=device, as_loss=False)
                metric_scores[m] = 0.0
            except Exception as e:
                print(f"Failed to load metric {m}: {e}")

    net_g = load_model(args, device)
    img_paths = sorted(glob.glob(os.path.join(args.input, '*')))
    print(f"🚀 Start Global-Resize Inference: {len(img_paths)} images")
    
    total_inference_time = 0.0 

    for i, img_path in enumerate(img_paths):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        img, h_scaled, w_scaled, h_orig, w_orig = pre_process_and_resize(img_path, device, args.max_size)

        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()

        with torch.inference_mode():
            # 1. Encoder
            enc_feats = net_g.vqgan.multiscale_encoder(img)
            enc_feats = enc_feats[::-1] 
            x = enc_feats[0] 
            feat_to_quant = net_g.vqgan.before_quant(x)
            
            # 2. Transformer
            logits = net_g.transformer.forward(feat_to_quant, return_embeds=False)
            out_tokens = logits.argmax(dim=2)
            b, _, h_enc, w_enc = feat_to_quant.shape
            z_quant = net_g.vqgan.quantize.get_codebook_entry(out_tokens.reshape(b, 1, h_enc, w_enc))
            
            # 3. Decode Prep
            x = net_g.vqgan.after_quant(z_quant)

            # 4. PISA Injection
            if hasattr(net_g.vqgan, 'retinex_adapter'):
                guidance = net_g.vqgan.retinex_adapter(img)
                if guidance.shape[-2:] != x.shape[-2:]:
                    guidance = F.interpolate(guidance, size=x.shape[-2:], mode='bilinear')
                x = x + guidance

            # 5. Decoder
            for depth_idx in range(net_g.max_depth):
                cur_res = net_g.gt_res // 2**net_g.max_depth * 2**depth_idx
                current_alpha = args.alpha if depth_idx == net_g.max_depth - 1 else 0.0
                
                if current_alpha > 0:
                    x = net_g.fuse_convs_dict[str(cur_res)](enc_feats[depth_idx].detach(), x, current_alpha)
                x = net_g.vqgan.decoder_group[depth_idx](x)
                
            output = net_g.vqgan.out_conv(x)
            
            output = output[..., :h_scaled, :w_scaled]
            if h_scaled != h_orig or w_scaled != w_orig:
                output = F.interpolate(output, size=(h_orig, w_orig), mode='bilinear', align_corners=False)

            output = adaptive_haze_correction(output)

        if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_inference_time += inference_time

        sr_img = tensor2img(output)
        save_path = os.path.join(args.output, f'{img_name}.png')
        imwrite(sr_img, save_path)
        
        log_str = f'[{i+1}/{len(img_paths)}] {img_name}: Time={inference_time:.4f}s'
        
        if args.metrics:
            sr_tensor = img2tensor(sr_img).unsqueeze(0).to(device) / 255.
            gt_tensor = None
            if args.gt:
                for ext in ['.png', '.jpg', '.jpeg']:
                    gt_p = os.path.join(args.gt, img_name + ext)
                    if os.path.exists(gt_p):
                        gt_img = cv2.imread(gt_p, cv2.IMREAD_COLOR)
                        gt_img = gt_img.astype(np.float32) / 255.
                        gt_tensor = torch.from_numpy(np.transpose(gt_img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)
                        break
            
            for m_name, func in metric_funcs.items():
                try:
                    if gt_tensor is not None: 
                        score = func(sr_tensor, gt_tensor).item()
                    else: 
                        score = func(sr_tensor).item()
                    metric_scores[m_name] += score
                    log_str += f' {m_name}={score:.4f}'
                except Exception as e:
                    pass
        print(log_str)

        del img, output, enc_feats, logits, z_quant, x, feat_to_quant, out_tokens
        if args.metrics:
            if 'sr_tensor' in locals(): del sr_tensor
            if 'gt_tensor' in locals(): del gt_tensor
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_time = total_inference_time / len(img_paths)
    print(f"\n====== Performance ======")
    print(f"Total Images: {len(img_paths)}")
    print(f"Avg Time: {avg_time:.4f} s/img") 

    if args.metrics:
        print("\n====== Metrics ======")
        for m, s in metric_scores.items():
            print(f"Average {m}: {s/len(img_paths):.4f}")

if __name__ == '__main__':
    main()