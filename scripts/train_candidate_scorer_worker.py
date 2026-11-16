"""One-GPU sequential head jobs; wait for shared visual caches when needed."""
import argparse
import time

import torch

from candidate_scorer_protocol import OUT
from point_aligned_data import read
from train_candidate_scorer import train


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seed', required=True, choices=['s1', 's2'])
    p.add_argument('--device', required=True)
    args = p.parse_args(); torch.set_num_threads(4)
    for variant in ('internal_full', 'internal_pooled', 'internal_global', 'visual'):
        if variant == 'visual':
            while not all((OUT / 'cache' / (s + '_index.json')).exists() and
                          read(OUT / 'cache' / (s + '_index.json')).get('visual_complete') for s in ('train', 'val')):
                time.sleep(5)
        train(variant, args.seed, args.device)
        torch.cuda.empty_cache()
