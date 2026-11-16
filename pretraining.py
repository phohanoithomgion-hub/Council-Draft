import os
import math
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CouncilModel
from dataset import StreamingPretrainDataset, PackedPretrainDataset, get_tokenizer
from config import get_config
from log import TrainingLogger
import torch

def set_vram_budget(fraction=0.9):
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    budget_bytes = int(total * fraction)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"[vram] total={total/1e9:.2f}GB budget={budget_bytes/1e9:.2f}GB fraction={fraction}")

def find_latest_pretrain_checkpoint(ckpt_dir):
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, "pretrain_curriculum_*.pt"))
    if not ckpts:
        ckpts = glob.glob(os.path.join(ckpt_dir, "pretrain_step_*.pt"))
    if not ckpts:
        return None, 0
    def get_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except Exception:
            return 0
    ckpts.sort(key=get_step, reverse=True)
    return ckpts[0], get_step(ckpts[0])


def warmup_cosine(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

def custom_collate_fn(batch):
    max_cu_len = max(len(x["cu_seqlens"]) for x in batch)
    for x in batch:
        cu = x["cu_seqlens"]
        pad_len = max_cu_len - len(cu)
        if pad_len > 0:
            x["cu_seqlens"] = torch.cat([cu, torch.full((pad_len,), cu[-1], dtype=cu.dtype)])
    return torch.utils.data.dataloader.default_collate(batch)


def train_pretrain(args_list=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_phase", type=int, default=None)
    args = parser.parse_args(args_list or [])

    config = get_config()
    device = torch.device(config.device)
    set_vram_budget(fraction=0.90)
    torch.manual_seed(config.seed)

    tokenizer = get_tokenizer()
    config.vocab_size = len(tokenizer)
    config.eos_token_id = tokenizer.eos_token_id
    print(f"Tokenizer vocab_size = {config.vocab_size}")

    logger = TrainingLogger(config.log_dir, "pretrain")
    logger.info(f"Config: {config}")

    n_rec = getattr(config, "pretrain_num_recursion", 2)
    total_steps = config.pretrain_max_steps
    save_interval = max(1, int(total_steps * config.pretrain_save_every_ratio))
    logger.info(f"Single-phase pretrain: n_rec={n_rec}, max_steps={total_steps}")

    if getattr(config, "pretrain_use_packing", False):
        dataset = PackedPretrainDataset(
            tokenizer,
            max_seq_len=config.max_seq_len,
            max_tokens_per_batch=config.pretrain_max_tokens_per_packed_batch,
            config_names=config.pretrain_dataset_configs,
        )
        print(f"[packing] enabled, max_tokens={config.pretrain_max_tokens_per_packed_batch}")
    else:
        dataset = StreamingPretrainDataset(
            tokenizer,
            max_seq_len=config.max_seq_len,
            config_names=config.pretrain_dataset_configs,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.pretrain_batch_size,
        num_workers=3,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        multiprocessing_context="spawn",
        collate_fn=custom_collate_fn if getattr(config, "pretrain_use_packing", False) else None,
    )

    config.backbone_checkpoint = True
    model = CouncilModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params:,}")

    resume_ckpt, _ = find_latest_pretrain_checkpoint(config.checkpoint_dir)
    resume_state = None
    step = 0
    total_real_tokens = 0
    if resume_ckpt is not None:
        logger.info(f"Found pretrain checkpoint: {resume_ckpt}")
        resume_state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        resume_state["model"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in resume_state["model"].items()
        }
        missing, unexpected = model.load_state_dict(resume_state["model"], strict=False)
        logger.info(f"Resumed model weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        step = resume_state.get("step", resume_state.get("total_step", 0))
        total_real_tokens = resume_state.get("total_real_tokens", 0)
        logger.info(f"Resumed step={step}, tokens={total_real_tokens/1e9:.3f}B")

    if step >= total_steps:
        logger.info(f"Already trained {step} >= {total_steps}. Nothing to do.")
        return model

    use_compile = os.environ.get("USE_COMPILE", "1") == "1"
    if use_compile:
        try:
            logger.info("Compiling model...")
            model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
            logger.info("Compile OK")
        except Exception as e:
            logger.info(f"Compile failed: {e}")

    beta2 = getattr(config, "pretrain_beta2", 0.99)
    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(), lr=config.pretrain_lr, betas=(0.9, beta2)
    )
    print(f"[optimizer] beta2={beta2}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: warmup_cosine(s, config.pretrain_warmup, total_steps),
    )
    if resume_state is not None:
        try:
            if "optimizer" in resume_state:
                optimizer.load_state_dict(resume_state["optimizer"])
            if "scheduler" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler"])
        except Exception as e:
            logger.info(f"Optimizer resume failed: {e}")

    model.train()
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Pretrain n_rec={n_rec}")
    for batch in pbar:
        if step >= total_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        n_tokens_tensor = batch["n_tokens"].sum()
        total_real_tokens += n_tokens_tensor

        cu_seqlens = batch.get("cu_seqlens", None)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device)
            if cu_seqlens.dim() == 1:
                cu_seqlens = cu_seqlens.unsqueeze(0)

        with autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(
                input_ids,
                labels=labels,
                num_recursion=n_rec,
                force_full_recursion=True,
                cu_seqlens=cu_seqlens,
            )
            loss = loss / config.pretrain_grad_accum

        loss.backward()

        if (step + 1) % config.pretrain_grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1

        if step % 10 == 0:
            current_loss = loss.item() * config.pretrain_grad_accum
            current_lr = scheduler.get_last_lr()[0]
            tokens_b = total_real_tokens.item() / 1e9 if isinstance(total_real_tokens, torch.Tensor) else total_real_tokens / 1e9
            pct = step / total_steps * 100
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "tokens": f"{tokens_b:.2f}B",
                "pct": f"{pct:.1f}%",
            })
            logger.log_step(step, {
                "loss": current_loss,
                "lr": current_lr,
                "tokens_B": tokens_b,
                "n_rec": n_rec,
            })

        if step % save_interval == 0 and step > 0:
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save({
                "model": save_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "total_real_tokens": total_real_tokens,
                "config": config,
            }, f"{config.checkpoint_dir}/pretrain_step_{step}.pt")
            logger.info(f"Saved pretrain_step_{step}.pt")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model": save_model.state_dict(),
        "step": step,
        "total_real_tokens": total_real_tokens,
        "config": config,
    }, f"{config.checkpoint_dir}/pretrain_final.pt")
    logger.info(f"Pretrain completed. step={step}, tokens={total_real_tokens/1e9:.3f}B")
    return model

if __name__ == "__main__":
    train_pretrain()