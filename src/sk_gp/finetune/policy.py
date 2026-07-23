"""QLoRA Scene-LLM policy (Section III fine-tuning).

A thin wrapper over a 4-bit quantized causal LM with LoRA adapters that maps a
scene-graph prompt to a JSON lengthscale-ratio configuration.  The backbone stays
frozen and quantized; only the low-rank adapter matrices ``theta`` are trained,
preserving the base model's reasoning while specializing it for semantics-to-kernel
translation.

Default LoRA hyperparameters follow the paper: rank ``r = 16``, scaling
``alpha = 32``, dropout ``0.05``, AdamW learning rate ``2e-4``, effective batch
size ``2 x 8`` (grad accumulation), ``5`` epochs, max sequence length ``4096``.
Adapters are inserted into the attention projections and the feed-forward
projections.  The HuggingFace stack (transformers + peft + bitsandbytes) is
imported lazily; if unavailable, :meth:`propose` returns the fallback config so the
reward-guided search still runs.

Requires the ``finetune`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

from ..scene_llm.config import sanitize_lengthscale_ratios
from ..scene_llm.prompts import LENGTHSCALE_SYSTEM_PROMPT
from .records import ReplayExample

__all__ = ["QLoRALengthscalePolicy", "DEFAULT_TARGET_MODULES"]

# Attention (q,k,v,o) + feed-forward (up,down) projections.
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def _extract_json_candidate(text: str) -> Dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
    return None


class _CompletionDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class QLoRALengthscalePolicy:
    def __init__(
        self,
        *,
        base_model: str = "Qwen/Qwen3-8B",
        adapter_source: str | None = None,
        trainable: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_seq_len: int = 4096,
        learning_rate: float = 2e-4,
        per_device_batch_size: int = 2,
        grad_accum_steps: int = 8,
        train_epochs: float = 5.0,
        target_modules: Sequence[str] | None = None,
    ):
        self.base_model = str(base_model)
        self.adapter_source = str(adapter_source) if adapter_source else None
        self.trainable = bool(trainable)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.max_seq_len = int(max_seq_len)
        self.learning_rate = float(learning_rate)
        self.per_device_batch_size = int(per_device_batch_size)
        self.grad_accum_steps = int(grad_accum_steps)
        self.train_epochs = float(train_epochs)
        self.target_modules = list(target_modules) if target_modules else list(DEFAULT_TARGET_MODULES)
        self._tokenizer = None
        self._model = None
        self._hf = None
        self._load_error: str | None = None
        self._quantized = False

    @property
    def is_available(self) -> bool:
        return self.ensure_loaded()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _import_hf(self) -> Dict[str, Any]:
        if self._hf is not None:
            return self._hf
        try:
            from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
        except Exception as exc:  # pragma: no cover - optional
            raise RuntimeError(
                "Missing HuggingFace fine-tuning stack. Install the finetune extra: "
                'pip install "transformers>=4.40" "peft>=0.10" "bitsandbytes>=0.43" "accelerate>=0.30"'
            ) from exc
        self._hf = {
            "LoraConfig": LoraConfig,
            "PeftModel": PeftModel,
            "get_peft_model": get_peft_model,
            "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "BitsAndBytesConfig": BitsAndBytesConfig,
            "Trainer": Trainer,
            "TrainingArguments": TrainingArguments,
        }
        return self._hf

    def _model_device(self):
        import torch

        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _render_prompt(self, prompt_text: str) -> str:
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": LENGTHSCALE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"<|system|>\n{LENGTHSCALE_SYSTEM_PROMPT}\n<|user|>\n{prompt_text}\n<|assistant|>\n"

    def ensure_loaded(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch

            hf = self._import_hf()
            tokenizer = hf["AutoTokenizer"].from_pretrained(self.base_model, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            quantized = False
            model_kwargs: Dict[str, Any] = {"device_map": "auto" if torch.cuda.is_available() else None}
            if torch.cuda.is_available():
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model_kwargs["quantization_config"] = hf["BitsAndBytesConfig"](
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                model_kwargs["torch_dtype"] = compute_dtype
                quantized = True

            model = hf["AutoModelForCausalLM"].from_pretrained(self.base_model, **model_kwargs)

            if self.adapter_source:
                model = hf["PeftModel"].from_pretrained(model, self.adapter_source, is_trainable=self.trainable)
            elif self.trainable:
                if quantized:
                    model = hf["prepare_model_for_kbit_training"](model)
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                lora_cfg = hf["LoraConfig"](
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=self.target_modules,
                )
                model = hf["get_peft_model"](model, lora_cfg)

            if self.trainable and hasattr(model, "config"):
                model.config.use_cache = False

            self._quantized = quantized
            self._tokenizer = tokenizer
            self._model = model
            return True
        except Exception as exc:  # pragma: no cover - optional
            self._load_error = str(exc)
            self._tokenizer = None
            self._model = None
            return False

    def propose(self, *, prompt_text: str, fallback_cfg: Dict[str, Any], temperature: float = 0.35, max_new_tokens: int = 180) -> Dict[str, Any]:
        """Sample one lengthscale-ratio configuration from the policy."""
        if not self.ensure_loaded():
            return dict(fallback_cfg)
        import torch

        prompt = self._render_prompt(prompt_text)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = self._model_device()
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                do_sample=True,
                temperature=float(max(temperature, 1e-3)),
                top_p=0.95,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[-1])
        text = self._tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        return sanitize_lengthscale_ratios(_extract_json_candidate(text), fallback=fallback_cfg)

    def _encode_supervised_row(self, prompt_text: str, answer_json: Dict[str, Any]) -> Dict[str, list]:
        prompt = self._render_prompt(prompt_text)
        answer = json.dumps(answer_json, separators=(",", ":")) + (self._tokenizer.eos_token or "")
        prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self._tokenizer(answer, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids[: min(len(answer_ids), self.max_seq_len)]
        max_prompt_len = max(self.max_seq_len - len(answer_ids), 0)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []
        input_ids = prompt_ids + answer_ids
        return {
            "input_ids": input_ids,
            "labels": ([-100] * len(prompt_ids)) + answer_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def _collate_batch(self, batch):
        import torch

        pad_id = int(self._tokenizer.pad_token_id)
        max_len = max(len(row["input_ids"]) for row in batch)
        out_ids, out_labels, out_mask = [], [], []
        for row in batch:
            pad = max_len - len(row["input_ids"])
            out_ids.append(row["input_ids"] + [pad_id] * pad)
            out_labels.append(row["labels"] + [-100] * pad)
            out_mask.append(row["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(out_ids, dtype=torch.long),
            "labels": torch.tensor(out_labels, dtype=torch.long),
            "attention_mask": torch.tensor(out_mask, dtype=torch.long),
        }

    def fine_tune(self, *, examples: Sequence[ReplayExample], save_dir: Path, max_examples: int) -> bool:
        """Distill the highest-reward candidates into the LoRA adapter.

        The replay examples are the positive (``k*``) candidates selected by the
        contrastive objective (Eq. 11); supervised completion training on them
        maximizes ``log pi_theta(k* | G_i)``.
        """
        if not self.trainable or not self.ensure_loaded() or not examples:
            return False
        import torch

        hf = self._import_hf()
        ranked = sorted(examples, key=lambda item: item.reward, reverse=True)[: int(max_examples)]
        dataset = _CompletionDataset([self._encode_supervised_row(e.prompt_text, e.answer_json) for e in ranked])
        training_args = hf["TrainingArguments"](
            output_dir=str(Path(save_dir) / "trainer_state"),
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.grad_accum_steps,
            num_train_epochs=self.train_epochs,
            learning_rate=self.learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit" if self._quantized else "adamw_torch",
        )
        trainer = hf["Trainer"](model=self._model, args=training_args, train_dataset=dataset, data_collator=self._collate_batch)
        trainer.train()
        self.save_adapter(Path(save_dir))
        return True

    def save_adapter(self, save_dir: Path) -> bool:
        if not self.ensure_loaded():
            return False
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(save_dir))
        self._tokenizer.save_pretrained(str(save_dir))
        return True
