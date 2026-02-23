# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by Henrique Morimitsu
# - Adapt code from JAX to PyTorch

"""Fast decoding routines for non-autoregressive generation."""

from typing import Callable
from einops import rearrange
import torch
import torch.nn.functional as F
import math
from basicsr.utils import mask_schedule
import numpy as np

def log(t, eps = 1e-10):
    return torch.log(t + eps)
def exists(val):
    return val is not None
def gumbel_noise(probs):
    noise = torch.zeros_like(probs).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(probs, temperature = 1., dim = -1):
    return ((probs / max(temperature, 1e-10)) + gumbel_noise(probs)).argmax(dim = dim)

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def sample_top_p(probs, p=0.75):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return probs_sort
    # next_token = torch.multinomial(probs_sort, num_samples=1)
    # next_token = torch.gather(probs_idx, -1, next_token)
    # return next_token

def mask_by_random_topk(
    mask_len: int,
    confidence: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Modifies from jax.random.choice without replacement.

    JAX's original implementation is as below:
        g = -gumbel(key, (n_inputs,)) - jnp.log(p)
        ind = jnp.argsort(g)[:n_draws]
    We adds temperature annealing on top of it, which is:
        g = -gumbel(key, (n_inputs,)) - temperature * jnp.log(p)
        ind = jnp.argsort(g)[:n_draws]

    Args:
        mask_len: the number to mask.
        probs: the probabilities associated with each entry.
        temperature: when temperature = 1.0, it's identical to jax's implementation.
        The larger this value is, the more random the masking is picked.

    Returns:
        A binary masking map [batch_size, seq_len].
    """

    g = torch.distributions.gumbel.Gumbel(0, 1)
    # confidence = torch.log(probs) + temperature * g.sample(probs.shape).to(probs.device)
    sorted_confidence = torch.sort(confidence, dim=-1,descending=True)[0]
    # Obtains cut off threshold given the mask lengths.
    cut_off = torch.gather(sorted_confidence, -1, mask_len)
    # Masks tokens with lower confidence.
    masking = (confidence >= cut_off)
    return masking


class State:
    """Holds decoding state data."""
    def __init__(
        self,
        cur_index: int,  # scalar int32: current decoded length index
        cur_seqs: torch.Tensor,  # int32 [batch, seq_len]
        final_seqs: torch.Tensor,  # int32 [batch, num_iter, seq_len],
        final_masks:torch.Tensor,  # int32 [batch, seq_len]
    ) -> None:
        self.cur_index = cur_index
        self.cur_seqs = cur_seqs
        self.final_seqs = final_seqs
        self.final_masks=final_masks


def state_init(
    init_indices: torch.Tensor,
    num_iter: int,
    start_iter: int = 0,
) -> State:
    """Initializes the decoding state data structure."""
    final_seqs0 = init_indices.unsqueeze(1)
    final_seqs0 = final_seqs0.repeat(1, num_iter, 1)

    return State(
        cur_index=start_iter, cur_seqs=init_indices, final_seqs=final_seqs0,final_masks=final_seqs0.clone())

def decode_critic(
    mask_tokens:torch.Tensor,
    lq_feats: torch.Tensor,
    tokens_to_logits: Callable[[torch.Tensor], torch.Tensor],
    tokens_to_feats:Callable[[torch.Tensor], torch.Tensor],
    mask_token_id: int = -1,
    num_iter: int = 12,
    start_iter: int = 0,
    choice_temperature: float = 4.5,
    mask_scheduling_method: str = "cosine",
    beta:float = 1.0
) -> torch.Tensor:
    b,c,h,w=lq_feats.shape

    hq_feats=(lq_feats).clone()
    # mask_feats=torch.zeros_like(lq_feats)
    mask_len=h*w

    unknown_number_in_the_beginning = torch.sum(mask_tokens == mask_token_id, dim=-1)
    # Initializes state
    state = state_init(mask_tokens, num_iter, start_iter=start_iter)

    # lq_tokens=inputs[...,inputs.shape[-1]//2:]
    hq_feats=None
    critic_probs=torch.zeros(1,num_iter,h*w).to(lq_feats.device)

    for step in range(start_iter, num_iter):
        """Beam search."""
        # Current input ids: [batch_size, seq_length].
        cur_ids = state.cur_seqs

        # Just updates the masked tokens.
        unknown_map = (cur_ids == mask_token_id)

        hq_feats=tokens_to_feats((cur_ids*~unknown_map).reshape(b,1,h,w))

        # mask=unknown_map.unsqueeze(1).repeat(1,256,1,1)
        mask=unknown_map.reshape(1,1,h,w).repeat(1,256,1,1)

        input_feats=hq_feats*~mask+lq_feats*mask
       
        # Calls model on current seqs to get next-iteration seqs.
        logits = tokens_to_logits(input_feats)
        # Computes the probabilities of each selected tokens.
        probs = F.softmax(logits, -1)
        # Samples the ids using categorical sampling: [batch_size, seq_length].
        b = probs.shape[0]

        sampled_ids = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)[..., 0]
       
        sampled_ids = rearrange(sampled_ids, '(b n) -> b n', b=b)
        # sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        
        # Updates final seqs with the current sampled_ids.
        state.final_seqs[:, step] = sampled_ids
        state.final_masks[:, step] = unknown_map
        
        selected_probs = torch.gather(probs, -1, sampled_ids.clamp(0, probs.shape[-1]-1).unsqueeze(-1)).squeeze(-1)
        selected_probs = torch.where(unknown_map, selected_probs,
                                    torch.zeros_like(selected_probs) + torch.inf)
        
        critic_logits = torch.sigmoid(tokens_to_logits(sampled_ids,h,w,True))
       
        critic_probs[:,step] = critic_logits #normalized_tensor #critic_logits
      
        
        # Defines the mask ratio for the next round. The number to mask out is determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter
        
        mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,mask_scheduling_method,beta)
        # Gets mask lens for each sample in the batch according to the mask ratio.
        # print(mask_ratio)
        mask_len = torch.unsqueeze(
            torch.floor(unknown_number_in_the_beginning *mask_ratio), 1)    

        mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(unknown_map, dim=-1, keepdim=True) - 1).long()
        # masking = critic_logits>0.5
        # Adds noise for randomness
        
        masking = mask_by_random_topk(mask_len, critic_logits,
                                    choice_temperature * (ratio))
        
        sampled_ids = torch.where(masking, mask_token_id, sampled_ids)
        
        state.cur_index += 1
        state.cur_seqs = sampled_ids
  
    return state.final_seqs,state.final_masks
