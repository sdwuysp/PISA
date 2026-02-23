import torch
from collections import OrderedDict
from os import path as osp
import os
from basicsr.utils import img2tensor, tensor2img, imwrite
from tqdm import tqdm
from basicsr.data.transforms import paired_random_crop
from basicsr.archs import build_network
from basicsr.losses import build_loss
from .base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torchvision.utils as tvu
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
from basicsr.utils.mask_schedule import schedule
import math
from ..utils.parallel_decode import gumbel_sample
import torch.nn as nn
import copy
import torch.nn as nn
import pyiqa

def _unwrap(m):
    return m.module if isinstance(m, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else m

@MODEL_REGISTRY.register()
class PredictorModel(BaseModel):
    """Base UnderwaterNet model for single image super-resolution."""
    def __init__(self, opt):
        super().__init__(opt)
        self.queue_size = opt.get('queue_size', 180)

        # define network g
        self.net_g = build_network(self.opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # define network hq
        self.net_hq = build_network(self.opt['network_hq'])
        self.net_hq = self.model_to_device(self.net_hq)
        self.print_network(self.net_hq)

        load_path = self.opt['path'].get('pretrain_network_hq', None)
        if load_path is not None:
            self.load_network(self.net_hq, load_path,
                              False)
            

        # define metric functions 
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items(): 
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # load pre-trained HQ ckpt, frozen decoder and codebook
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            self.load_network(_unwrap(self.net_g).vqgan, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get(
                'frozen_module_keywords', None)
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            # print(fkw)
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0) 
            self.net_d_best = copy.deepcopy(self.net_d)
        
        self.net_g_best = copy.deepcopy(self.net_g)

    def feed_data(self, data):
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        
        # [Fix] Explicitly handle missing GT to avoid stale data from previous training batches
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = None # Clear stale data
       
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            # [v1.0] Only optimize parameters that require grad (Adapter & Decoder)
            if v.requires_grad:
                optim_params.append(v)
            else:
                pass
                # logger = get_root_logger()
                # logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr +
                          b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr +
                          b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    def init_training_settings(self):
        self.net_hq.eval()
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
            
        self.net_d.train()
        train_opt = self.opt['train']

        # define losses

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        
        if self.cri_pix is None and self.cri_perceptual is None and self.cri_feats is None:
            print('Both pixel, perceptual, and feats losses are None.')
            pass
            # raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        # [v1.0 Safety Check] Print active training parameters
        logger.info("====== [PISA Strategy] Active Training Parameters ======")
        count = 0
        for name, param in self.net_g.named_parameters():
            if param.requires_grad:
                logger.info(f"Training: {name}")
                count += 1
        logger.info(f"Total active parameter groups: {count}")

        self.ema_decay = self.opt['train'].get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = copy.deepcopy(self.net_g)
            self.net_g_ema.eval()
            self.net_g_ema = self.model_to_device(self.net_g_ema)

    def get_current_visuals(self):
        vis_samples = 4
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]


        if hasattr(self, 'output'):
            out_dict['output'] = self.output.detach().cpu()[:vis_samples]
            # cv2.imwrite("output.jpg", tensor2img(self.output[0]))
        if hasattr(self, 'gt') and self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        if hasattr(self, 'hq_rec'):
            out_dict['hq_rec'] = self.hq_rec.detach().cpu()[:vis_samples]


        return out_dict

    def optimize_parameters(self, current_iter):

        train_opt = self.opt['train']
        code_only=self.opt['network_g']['code_only']
        detach_16=self.opt['network_g']['detach_16']

        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        if self.LQ_stage:
            with torch.no_grad():
                self.hq_rec, _, _, gt_indices = self.net_hq(self.gt)
                # print("gt_indices:",gt_indices.shape)
                b, _, h, w = gt_indices.shape
                target_feats = _unwrap(self.net_hq).quantize.get_codebook_entry(
            gt_indices)
        
        # mask
        r = np.random.uniform()
        r = math.floor(schedule(r, h * w) * h * w)
        
        samples = torch.rand(b,h * w,dtype=torch.float,device=self.device).topk(r,dim=1).indices
       
        token_mask = torch.zeros(b,h * w,device=self.device,dtype=torch.bool)
        token_mask.scatter_(dim=1, index=samples, value=True)
        token_mask = token_mask.reshape(b, 1, h, w)
        
        # [v1.0] self.lq is passed, net_g will automatically use it for Adapter guidance
        logits, lq_feats,self.output = self.net_g(self.lq, token_mask=token_mask,hq_feats=target_feats,
                alpha=1,code_only=code_only,detach_16=detach_16)  # logits (hw)bn -> b(hw)n

        l_g_total = 0
        loss_dict = OrderedDict()
        
        # MSE loss feats alain codebook
        l_feat_encoder = torch.mean((target_feats.detach()-lq_feats)**2)
        l_g_total += l_feat_encoder
        loss_dict['l_feat_encoder'] = l_feat_encoder
    
        # cross_entropy loss  b(hw)n -> bn(hw)
        l_token = F.cross_entropy(logits.permute(0, 2, 1),
                                         gt_indices.flatten(1))*train_opt['cross_entropy_opt']['token_weight']
        
        loss_dict['l_token'] = l_token
        l_g_total += l_token

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style
        
        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.mean().backward()
        self.optimizer_g.step()
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()
 
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            lq = self.lq
            gt = getattr(self, 'gt', None)
            
            B, C, H, W = lq.shape
            window_size = 8
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            if pad_h != 0 or pad_w != 0:
                lq = F.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')
                if gt is not None:
                    gt = F.pad(gt, (0, pad_w, 0, pad_h), mode='reflect')

            code_only = self.opt['network_g'].get('code_only', False)
            detach_16 = self.opt['network_g'].get('detach_16', False)

            if self.LQ_stage:
                hq_input = lq 
                self.hq_rec, _, _, gt_indices = self.net_hq(hq_input)
                B2, _, H2, W2 = gt_indices.shape
                target_feats = _unwrap(self.net_hq).quantize.get_codebook_entry(gt_indices)
                token_mask = torch.ones(B2, H2 * W2, device=self.device, dtype=torch.bool).view(B2, 1, H2, W2)

                n_iter = self.opt['val'].get('n_iter', 1) 
                current_input = lq
                
                for i in range(n_iter):
                    _, _, out = self.net_g(
                        current_input,
                        token_mask=token_mask,
                        hq_feats=target_feats,
                        alpha=1,
                        code_only=code_only,
                        detach_16=detach_16,
                        adapter_img=lq
                    )
                    current_input = out
                
                self.output = out
            else:
                self.output = _unwrap(self.net_g)(lq)

            if pad_h != 0 or pad_w != 0:
                self.output = self.output[..., :H, :W]
                if hasattr(self, 'hq_rec'):
                    self.hq_rec = self.hq_rec[..., :H, :W]
                if gt is not None:
                    self.gt = gt[..., :H, :W]
            else:
                if gt is not None:
                    self.gt = gt

        self.net_g.train()

        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            
        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric') 

        nr_metrics = ['clipiqa', 'musiq', 'maniqa', 'niqe', 'brisque', 'pi', 'fid']

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            
            sr_img = tensor2img(self.output)
            
            sr_tensor = img2tensor(sr_img).unsqueeze(0).to(self.device) / 255.

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}', 
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    try:
                        if name.lower() in nr_metrics:
                            tmp_result = self.metric_funcs[name](sr_tensor)
                            self.metric_results[name] += tmp_result.item()

                        elif hasattr(self, 'gt') and self.gt is not None:
                            tmp_result = self.metric_funcs[name](sr_tensor, self.gt)
                            self.metric_results[name] += tmp_result.item()
                        
                        else:
                            continue
                            
                    except Exception as e:
                        pass

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()
            
        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            
            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric, self.metric_results[self.key_metric], current_iter)
            
                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')
            else:
                # update each metric separately 
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated 
                if sum(updated): 
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if hasattr(self, 'net_g_ema'):
            self.save_network(self.net_g_ema, 'net_g_ema', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)