# FNC — From Naming to Clicking

### GUI Grounding with Spatially Aligned Internal Evidence

FNC maps a natural-language instruction and a GUI screenshot to a target click location. It uses a frozen vision-language model to generate a target description, reads internal evidence at candidate locations, and learns a lightweight scoring function to select the final point.

## Method

FNC connects target naming with spatial selection through three components:

1. **Target naming and observation.** A frozen Qwen3-VL backbone generates target descriptions from the screenshot and local views.
2. **Candidate-aligned evidence.** A trained proposer produces candidate locations. Attention responses and hidden-state features are sampled at each candidate's corresponding spatial position.
3. **Learned readout.** A lightweight scorer aggregates the evidence across tokens and views, ranks the candidates, and returns the highest-scoring click location.

The backbone remains frozen. The candidate proposer and evidence readout are learned modules.

```text
Screenshot + Instruction
          |
          v
Frozen VLM: target descriptions and internal responses
          |
          v
Candidate locations + spatially aligned evidence
          |
          v
Lightweight readout -> Click location
```

## Code Overview

| Component | Source |
| --- | --- |
| Target description prompts and token handling | [target_token.py](src/clickagent/target_token.py) |
| Target-conditioned feature extraction | [conditioned_features.py](src/clickagent/conditioned_features.py) |
| Multi-region candidate construction | [multiregion_proposals.py](src/clickagent/multiregion_proposals.py) |
| Candidate evidence aggregation and scoring | [evidence_retention.py](src/clickagent/evidence_retention.py) |
| Training, inference, and evaluation scripts | [scripts/](scripts/) |

The command-line entry point is `fnc`. The Python package remains `clickagent`, and the previous command name is retained as a compatibility alias.

## Installation

Use Python 3.10–3.12. The model dependencies are specified in [pyproject.toml](pyproject.toml), including Transformers 4.57.1 and PyTorch 2.6–2.7.

```bash
git clone https://github.com/DiWu17/FNC.git
cd FNC
python -m pip install -e ".[model]"
```

For GPU inference, install a PyTorch build compatible with your CUDA environment.

## Model and Data Setup

Place the Qwen3-VL-8B-Instruct model files under:

```text
models/Qwen3-VL-8B-Instruct/
```

Check the environment and model files with:

```bash
fnc doctor --model models/Qwen3-VL-8B-Instruct
```

End-to-end FNC inference also requires the trained candidate proposer and readout checkpoints. Model weights, datasets, and generated feature caches are managed separately from this source repository. Experiment scripts specify their required input files and output locations.

## Command-Line Utilities

Inspect the available commands:

```bash
fnc --help
```
