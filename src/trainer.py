"""
trainer.py (v9 - OFT + Cognitive Error Contrastive Signal, corrected)
----------------------------------------------------------------------
Two training signals:
  1. SFT loss on (prompt, gold) — same as static SFT baseline
  2. Contrastive margin loss on (wrong_output, gold) pairs — FMACA only

Key design decisions (following ChatGPT/Claude review):
  - wrong_output  = REAL model-generated output (not gold)
  - chosen_output = gold answer (safe, always correct, no hallucination)
  - No frozen reference model needed (margin loss, not full DPO)
  - train_step_only() + generate_outputs() separated so critic sees real outputs
  - _ensure_contrastive_buffer() initialises buffer lazily
"""

import os
import gc
import logging
from typing import List, Dict, Optional

import torch

logger = logging.getLogger(__name__)

MODEL_PATH = "${MODELS_DIR}/Qwen2.5-7B-Instruct"

OFT_CONFIG = {
    "oft_block_size": 16,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "module_dropout": 0.0,
    "init_weights": True,
}

TRAIN_CONFIG = {
    "learning_rate": 2e-5,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "contrastive_lambda": 0.3,
    "contrastive_margin": 0.5,
}


class FMACATrainer:
    """
    Qwen2.5-7B-Instruct with OFT adapter.

    For FMACA condition: SFT loss + contrastive loss on error pairs.
    For other conditions: SFT loss only (use_contrastive=False).
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        use_deepspeed: bool = False,
        output_dir: str = "./trainer_output",
        use_contrastive: bool = False,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.model_path      = model_path
        self.output_dir      = output_dir
        self.use_contrastive = use_contrastive
        self._current_mode   = None
        self._model          = None
        self._tokenizer      = None

        os.makedirs(output_dir, exist_ok=True)
        self._load_base_model()

    # ------------------------------------------------------------------
    # Contrastive buffer management
    # ------------------------------------------------------------------

    def _ensure_contrastive_buffer(self):
        """Lazily initialise contrastive buffer."""
        if not hasattr(self, "_contrastive_buffer"):
            self._contrastive_buffer = []
            logger.info("Contrastive buffer initialised.")

    def add_contrastive_pair(
        self,
        prompt: str,
        wrong_output: str,
        chosen_output: str,
        error_type: str,
        example_id: str,
    ) -> None:
        """
        Add (wrong_output, chosen_output) contrastive pair.

        wrong_output  = real model output containing cognitive error
        chosen_output = gold answer (safe, correct, no hallucination)
        error_type    = critic-detected cognitive error type
        """
        if not self.use_contrastive:
            return
        if error_type == "none":
            return
        if not wrong_output.strip() or not chosen_output.strip():
            return
        if wrong_output.strip() == chosen_output.strip():
            return  # no contrast if identical

        self._ensure_contrastive_buffer()

        self._contrastive_buffer.append({
            "prompt":     prompt,
            "wrong":      wrong_output,
            "chosen":     chosen_output,
            "error_type": error_type,
            "example_id": example_id,
        })

        # Keep buffer bounded
        if len(self._contrastive_buffer) > 200:
            self._contrastive_buffer.pop(0)

        logger.info(
            "Contrastive pair | error=%s | buffer=%d",
            error_type, len(self._contrastive_buffer),
        )

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def set_adapter(self, adapter_mode: str) -> None:
        """Apply OFT (or LoRA for GPT2-style models). Subsequent calls ignored."""
        if self._current_mode is not None:
            logger.debug("Adapter already set to %s - keeping.", self._current_mode)
            return

        from peft import get_peft_model, OFTConfig, LoraConfig, prepare_model_for_kbit_training

        arch = getattr(self._model.config, "architectures", [""])[0]                if hasattr(self._model, "config") else ""
        is_gpt2_style = "GPT2" in arch

        self._model = prepare_model_for_kbit_training(
            self._model, use_gradient_checkpointing=True
        )

        if os.environ.get("FMACA_ADAPTER", "oft") == "lora" and not is_gpt2_style:
            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                target_modules=OFT_CONFIG["target_modules"], task_type="CAUSAL_LM",
            )
            logger.info("Applying LoRA (FMACA_ADAPTER=lora) | r=16")
            self._model = get_peft_model(self._model, peft_config)
            self._current_mode = "lora"
            self.is_gpt2 = False
        elif is_gpt2_style:
            peft_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.05, bias="none",
            )
            logger.info("Applying LoRA (GPT2 arch) | r=8")
            self._model = get_peft_model(self._model, peft_config)
            self._current_mode = "lora"
            self.is_gpt2 = True
            logger.info("LoRA adapter applied.")
        else:
            peft_config = OFTConfig(
                oft_block_size=OFT_CONFIG["oft_block_size"],
                target_modules=OFT_CONFIG["target_modules"],
                module_dropout=OFT_CONFIG["module_dropout"],
                init_weights=OFT_CONFIG["init_weights"],
            )
            logger.info("Applying OFT | block_size=%d", OFT_CONFIG["oft_block_size"])
            self._model = get_peft_model(self._model, peft_config)
            self._current_mode = "oft"
            self.is_gpt2 = False
            logger.info("OFT adapter applied.")
        self._model.print_trainable_parameters()

    def train_and_generate(
        self,
        examples: List[Dict],
        prompts: List[str],
        adapter_mode: str = "oft",
    ) -> List[str]:
        """
        Legacy API: train + return gold outputs.
        Used by conditions that don't need real outputs for critic.
        """
        self.train_step_only(examples, prompts, adapter_mode)
        return [ex.get("output", "") for ex in examples]

    def train_step_only(self, examples, prompts, adapter_mode="oft"):
        """Manual AdamW SFT — no HF Trainer, no DeepSpeed conflicts."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        self.set_adapter(adapter_mode)
        self._ensure_contrastive_buffer()

        if not hasattr(self, "_sft_optimizer") or self._sft_optimizer is None:
            self._sft_optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self._model.parameters()),
                lr=float(os.environ.get("FMACA_LR", "2e-5")), weight_decay=0.01,
            )
            logger.info("AdamW optimizer created | lr=%s", os.environ.get("FMACA_LR", "2e-5"))

        train_texts = [p + "\n" + ex.get("output","") for p, ex in zip(prompts, examples)]
        enc = self._tokenizer(train_texts, return_tensors="pt",
                              padding=True, truncation=True, max_length=512)
        device    = next(self._model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        labels    = input_ids.clone()
        for i, p in enumerate(prompts):
            plen = self._tokenizer(p + "\n", return_tensors="pt",
                                   truncation=True, max_length=512)["input_ids"].shape[1]
            labels[i, :plen] = -100

        supervised = int((labels != -100).sum())
        if supervised == 0:
            logger.warning("No supervised tokens — skipping.")
            return 0.0
        logger.info("Supervised tokens: %d / %d", supervised, labels.numel())

        self._model.train()
        self._sft_optimizer.zero_grad()
        out = self._model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._sft_optimizer.step()
        self._sft_optimizer.zero_grad(set_to_none=True)  # revision: no SFT gradient leaks into contrastive step
        sft_loss = float(out.loss.detach().cpu())

        logger.info("SFT step | adapter=%s | sft_loss=%.4f | n=%d",
                    self._current_mode, sft_loss, len(examples))

        if not hasattr(self, "_sft_step_count"): self._sft_step_count = 0
        self._sft_step_count += 1
        if self.use_contrastive and len(self._contrastive_buffer) >= 2                 and self._sft_step_count % 10 == 0:
            self._contrastive_step()
        return sft_loss


    def _contrastive_step(self) -> None:
        """
        Contrastive margin loss on buffered (wrong, chosen) pairs.

        Loss = mean(max(0, margin - [loss_wrong - loss_chosen]))

        Pushes model to assign:
          HIGHER loss to wrong outputs (contain cognitive errors)
          LOWER  loss to chosen outputs (gold answers)
        """
        import random

        self._ensure_contrastive_buffer()
        if len(self._contrastive_buffer) < 2:
            return

        recent = self._contrastive_buffer[-min(20, len(self._contrastive_buffer)):]
        pairs  = random.sample(recent, min(4, len(recent)))
        device = next(self._model.parameters()).device
        margin = TRAIN_CONFIG["contrastive_margin"]

        total_loss = None
        valid = 0

        self._model.train()
        for pair in pairs:
            try:
                prompt_len = self._tokenizer(
                    pair["prompt"] + "\n",
                    return_tensors="pt", truncation=True, max_length=512,
                )["input_ids"].shape[1]

                def encode(text):
                    enc = self._tokenizer(
                        text, return_tensors="pt",
                        truncation=True, max_length=512,
                    )
                    lbl = enc["input_ids"].clone()
                    lbl[0, :prompt_len] = -100
                    return enc, lbl

                wrong_enc,  wrong_lbl  = encode(pair["prompt"] + "\n" + pair["wrong"])
                chosen_enc, chosen_lbl = encode(pair["prompt"] + "\n" + pair["chosen"])

                if (wrong_lbl  != -100).sum() == 0: continue
                if (chosen_lbl != -100).sum() == 0: continue

                out_wrong  = self._model(
                    input_ids=wrong_enc["input_ids"].to(device),
                    attention_mask=wrong_enc["attention_mask"].to(device),
                    labels=wrong_lbl.to(device),
                )
                out_chosen = self._model(
                    input_ids=chosen_enc["input_ids"].to(device),
                    attention_mask=chosen_enc["attention_mask"].to(device),
                    labels=chosen_lbl.to(device),
                )

                if os.environ.get("CONTRASTIVE_LOSS", "margin") == "dpo":
                    # sequence log-probabilities (sum over target tokens); reference = adapter disabled
                    n_w = int((wrong_lbl[:, 1:] != -100).sum()); n_c = int((chosen_lbl[:, 1:] != -100).sum())
                    lp_w, lp_c = -out_wrong.loss * n_w, -out_chosen.loss * n_c
                    with torch.no_grad(), self._model.disable_adapter():
                        ref_w = -self._model(input_ids=wrong_enc["input_ids"].to(device),
                                             attention_mask=wrong_enc["attention_mask"].to(device),
                                             labels=wrong_lbl.to(device)).loss * n_w
                        ref_c = -self._model(input_ids=chosen_enc["input_ids"].to(device),
                                             attention_mask=chosen_enc["attention_mask"].to(device),
                                             labels=chosen_lbl.to(device)).loss * n_c
                    beta = float(os.environ.get("DPO_BETA", "0.1"))
                    pair_loss = -torch.nn.functional.logsigmoid(beta * ((lp_c - ref_c) - (lp_w - ref_w)))
                else:
                    pair_loss = torch.clamp(
                        margin - (out_wrong.loss - out_chosen.loss),
                        min=0.0,
                    )
                total_loss = pair_loss if total_loss is None else total_loss + pair_loss
                valid += 1

            except Exception as e:
                logger.warning("Contrastive pair failed: %s", e)
                continue

        if total_loss is not None and valid > 0:
            c_loss = total_loss / valid
            # Manual backward for contrastive (separate from SFT Trainer step)
            (TRAIN_CONFIG["contrastive_lambda"] * c_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                TRAIN_CONFIG["max_grad_norm"],
            )
            # Use a simple optimizer for contrastive step
            if not hasattr(self, "_c_optimizer"):
                self._c_optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self._model.parameters()),
                    lr=float(os.environ.get("FMACA_LR", TRAIN_CONFIG["learning_rate"])),
                )
            self._c_optimizer.step()
            self._c_optimizer.zero_grad()

            logger.info(
                "Contrastive step | loss=%.4f | pairs=%d | buffer=%d",
                c_loss.item(), valid, len(self._contrastive_buffer),
            )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_outputs(self, prompts: List[str]) -> List[str]:
        """
        Generate REAL model outputs for critic evaluation.
        Short generation (64 tokens) for speed during training rounds.
        """
        self._model.eval()
        outputs = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = self._tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=768,
                ).to(next(self._model.parameters()).device)
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
                new_tokens = generated[0][inputs["input_ids"].shape[1]:]
                outputs.append(
                    self._tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                )
        return outputs

    def generate_only(self, prompts: List[str]) -> List[str]:
        """Full generation for evaluation (256 tokens)."""
        self._model.eval()
        outputs = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = self._tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=768,
                ).to(next(self._model.parameters()).device)
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
                new_tokens = generated[0][inputs["input_ids"].shape[1]:]
                outputs.append(
                    self._tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                )
        return outputs

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_base_model(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info("Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading Qwen2.5-7B-Instruct in 4-bit NF4 on rank %d...", local_rank)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self._model.config.use_cache = False
        self._model.enable_input_require_grads()
        logger.info(
            "Model loaded | params=%.1fB | device=cuda:%d",
            sum(p.numel() for p in self._model.parameters()) / 1e9,
            local_rank,
        )

    def cleanup(self) -> None:
        del self._model
        del self._tokenizer
        self._model = self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Trainer cleaned up.")