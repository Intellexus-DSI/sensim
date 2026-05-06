# SenSim — Sentence Similarity for Classical Tibetan

A pipeline for training and evaluating sentence similarity models for Classical Tibetan, using automatically generated annotations via Best-Worst Scaling (BWS) and large language models.

## Overview

This repository provides tools for:

- **Candidate pair generation** — building pools of sentence pairs from Classical Tibetan corpora
- **Automatic annotation** — scoring pairs using BWS-based annotation and LLM evaluators
- **Model training** — fine-tuning SBERT and cross-encoder models with contrastive and relative ranking losses
- **Active sampling** — iterative progressive learning to focus annotation effort on informative examples
- **Evaluation** — correlation-based benchmarking against human judgments

## Installation

```bash
pip install -r requirements.txt
```

Copy and fill in the configuration templates before running:

```bash
cp config.template.yaml config.yaml
cp keys.template.yaml keys.yaml
```

## Usage

**Training a sentence similarity model:**

```bash
bash sensim_eval_sbert.sh
```

**Running grid search:**

```bash
bash sensim_eval_grid_search.sh
```

**Active sampling pipeline:**

```bash
python active_sampling_pipeline.py
```

**Generating synthetic sentences:**

```bash
python generate_synthetic_sentences.py
```

## Citation

If you use this code or the associated data in your research, please cite:

```bibtex
@inproceedings{cohen2026scaling,
  title     = {Scaling Sentence Similarity for Classical Tibetan with Automatic Annotations},
  author    = {Cohen, Shay et al.},
  year      = {2026}
}
```

> **Note:** Please update the BibTeX entry above with the complete author list, venue, and publication details once the paper is published.

## License

Copyright 2025 Intellexus

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this repository except in compliance with the License. You may obtain a copy of the License at:

```
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
