"""Cache extra visual states, then train the second seed on the same GPU."""
import torch

from visual_candidate_cache import cache_visual_split
from train_candidate_scorer import train


if __name__ == '__main__':
    torch.set_num_threads(4)
    cache_visual_split('train', 'cuda:1')
    torch.cuda.empty_cache()
    cache_visual_split('val', 'cuda:1')
    torch.cuda.empty_cache()
    for variant in ('internal_full', 'internal_pooled', 'internal_global', 'visual'):
        train(variant, 's2', 'cuda:1')
        torch.cuda.empty_cache()
