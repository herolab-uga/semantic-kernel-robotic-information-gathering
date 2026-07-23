# Models & credentials

## Scene-LLM backbones

The paper evaluates three backbones (few-shot variant, FS) and fine-tunes the
smallest (FT):

| Tier | Model | Params | Used for |
|------|-------|--------|----------|
| [L]  | GPT-4.1 | ~1800B | upper-bound few-shot reasoning (hosted API) |
| [M]  | Llama-3.3 | 70B | mid-sized few-shot (hosted / local) |
| [S]  | Qwen3-8B | 8B | few-shot **and** the QLoRA fine-tuned (FT) variant |

The hosted few-shot (FS) path uses the curated exemplars in
`sk_gp.scene_llm.exemplars` (20 valid + 5 counter-examples, included). Put your key
in a git-ignored `api_key.txt` — never commit it:

```bash
printf 'YOUR_OPENAI_API_KEY' > api_key.txt   # git-ignored
```

The pre-trained fine-tuned (FT) adapter is not distributed; train your own with the
pipeline below.

Local models are downloaded automatically by `transformers` on first use, e.g.:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
```

Gated models (Llama) require `huggingface-cli login` and access approval.

## QLoRA requirements

Fine-tuning the 8B model with 4-bit NF4 quantization needs a CUDA GPU with
~16 GB VRAM and the `finetune` extra:

```bash
pip install -e ".[finetune]"
```

Fine-tuned adapters are written to `runs/finetune/saved_model/` and can be loaded
via `QLoRALengthscalePolicy(adapter_source=..., trainable=False)`.
