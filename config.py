import os
import math
from dataclasses import dataclass

@dataclass
class CouncilConfig:
    # Model
    d_model_backbone: int = 640
    num_blocks: int = 8
    num_streams: int = 4
    sparse_num_blocks: int = 32
    sparse_top_k: int = 6
    vocab_size: int = 50257  # Sẽ được ghi đè bằng len(tokenizer)
    d_model_thought: int = 1024
    num_qubits: int = 6
    quantum_num_layers: int = 2
    moe_num_prototypes: int = 8
    dynamic_hidden: int = 16000
    multidim_base: int = 2048
    engram_table_size: int = 50000
    engram_mem_dim: int = 64
    max_recursion: int = 1
    max_seq_len: int = 416
    pretrain_max_tokens_per_packed_batch: int = 416
    sliding_window: int = 416
    fixed_recursion: int = 1
    adaptive_inference: bool = False

    # SFT
    sft_batch_size: int = 16
    sft_grad_accum: int = 8
    sft_lr: float = 1e-4
    sft_warmup: int = 2000
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    sft_epochs: int = 1
    sft_max_tokens: int = 50_000_000_000
    sft_total_steps: int = 20000
    save_every_ratio: float = 1
    sft_fixed_recursion: int = 2

    # Stability & Initialization
    rezero_alpha_init: float = 1.0
    layerscale_gamma_init: float = 1.0
    deepnorm_alpha: float = 1.0
    gradient_clip_norm: float = 1.0

    # Checkpointing
    backbone_checkpoint: bool = True
    attention_checkpoint: bool = True

    # Pretraining
    pretrain_batch_size: int = 16
    pretrain_grad_accum: int = 2
    pretrain_lr: float = 3e-4
    pretrain_lr_warmdown_steps: int = 5
    pretrain_warmup: int = 500
    pretrain_max_tokens: int = 30_000_000_000_000
    pretrain_max_steps: int = 1_500_000
    pretrain_save_every_ratio: float = 0.02
    pretrain_use_packing: bool = True
    pretrain_beta2: float = 0.9968
    pretrain_dataset_configs: list = None
    pretrain_num_recursion: int = 1
    pretrain_curriculum_phases: list = None
    pretrain_steps_per_phase: list = None

    # RL (OPPO)
    rl_batch_size: int = 8
    rl_grad_accum: int = 16
    rl_lr: float = 2e-5
    rl_epochs: int = 1
    rl_max_tokens: int = 50_000_000_000
    rl_total_steps: int = 90000
    rl_clip_epsilon: float = 0.2
    rl_beta: float = 0.01
    rl_max_new_tokens: int = 256
    sample_every_n_steps: int = 500
    rl_temperature: float = 0.8
    rl_generate_recursion: int = 2
    rl_force_recursion: int = 2
    rl_format_weight: float = 0.3
    rl_accuracy_weight: float = 1.0
    rl_log_every_n_samples: int = 10
    rl_thinking_min_chars: int = 20
    use_cache: bool = True
    mamba_ssm_cache_dtype: str = "bfloat16"
    mamba_conv_cache_dtype: str = "bfloat16"

    # OPPO
    oppo_beta0_prior: float = 0.5
    oppo_advantage_scale: float = 4.0

    # Adaptive recursion
    aux_cont_loss_weight: float = 0.05

    # Data
    max_samples_per_dataset: int = 2_000_000
    num_workers: int = 0

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42

    # Inference generation
    inference_repetition_penalty: float = 1.3
    inference_no_repeat_ngram_size: int = 5

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

def get_config():
    config = CouncilConfig()
    if config.pretrain_dataset_configs is None:
        config.pretrain_dataset_configs = ["xerv_pure", "automathtext-v2-ultra"]
    return config