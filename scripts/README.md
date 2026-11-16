# Script Guide

Run scripts from the repository root after installing the package. This directory contains the feature extraction, proposer training, readout training, and evaluation code, together with their shared dependencies.

## Main entry points

| Task | Script | Required inputs |
| --- | --- | --- |
| Download backbone weights | `download_model.py` | Model identifier and output directory |
| Download and normalize supported public datasets | `download_public_data.py` | Dataset selection and data directory |
| Extract target-conditioned features | `extract_conditioned_features.py` | Model weights, sample manifest, and extraction configuration |
| Prepare proposer training records | `prepare_crossfit_candidates.py` | Prepared feature indexes and sample manifests |
| Train the full or cross-fit proposer | `train_crossfit_proposer.py` | Training/validation records, raw features, and fold assignments |
| Extract multi-region features | `extract_multiregion_v13.py` | Frozen proposer, backbone weights, and prepared records |
| Train the full-data readout | `run_full_scaling_v18.py` | Prepared training manifests, feature indexes, and registered configuration |
| Train readout ablations | `run_paper_scale_ablation_train_v1.py` | Cached features and registered training/model metadata |
| Extract, score, and evaluate registered cohorts | `run_paper_scale_eval_v1.py` | Cohort manifests, frozen checkpoints, feature caches, and evaluation metadata |

Inspect an entry point's arguments before running it, for example:

```bash
python scripts/download_model.py --help
python scripts/train_crossfit_proposer.py --help
python scripts/run_paper_scale_ablation_train_v1.py --help
```

## Supporting code

The remaining scripts provide data preparation, readout utilities, and dependencies shared with the optional verifier. In particular, `run_evidence_retention_v14.py`, `run_quality_readout_v15.py`, and `run_quality_readout_v15_3k.py` supply helpers reused by the full-data and ablation entry points. Their versioned filenames are retained to preserve imports and recorded source identities.

The evaluation runner separates prediction generation from label-based analysis. The training and evaluation scripts use explicit input manifests and source/checkpoint hashes; their default paths refer to prepared experiment artifacts. Weights, feature caches, fold maps, cohort manifests, and registration metadata are distributed separately. A source checkout alone does not contain those artifacts or reproduce the reported tables without them.
