# Models

Fine-tuned models are not stored in git. Download them from the
[GitHub release](https://github.com/gorzalniksamuel/EmbeddingsBench/releases) and unpack them here:

```bash
tar -xzf finetuned-gnad10.tar.gz -C models/
```

Each `models/<dataset>/` folder contains

| File | Content |
|---|---|
| `adapter/` | LoRA adapter (rank 16) for `jinaai/jina-embeddings-v5-text-small-classification`, revision `4447914a` |
| `head.pt` | Linear classification head on the L2-normalised last-token embedding |
| `classifier.json` | Labels, readable label names, prompt, input length, validation and test accuracy, training settings |

Load one with `load_finetuned()` from `bench/finetune.py` (see the main README).
Licence: CC BY-NC 4.0, as the base model.

Models trained through the registry live in `registry/models/<name>/v<N>/` (scikit-learn models are
stored with joblib; only load files you trust, since joblib uses pickle).
