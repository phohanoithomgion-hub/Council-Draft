import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

try:
    _COMPILED_FLEX_ATTENTION = torch.compile(
        flex_attention,
        dynamic=True,
        fullgraph=True,
        mode="default",
    )
except Exception:
    _COMPILED_FLEX_ATTENTION = flex_attention


class MLAAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_c=512, d_rotate=64, window_size=4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_c = d_c
        self.d_rotate = d_rotate
        self.window_size = window_size

        self.q_down = nn.Linear(d_model, d_c + d_rotate, bias=False)
        self.kv_down = nn.Linear(d_model, d_c + d_rotate, bias=False)
        self.q_up = nn.Linear(d_c, d_model, bias=False)
        self.k_up = nn.Linear(d_c, d_model, bias=False)
        self.v_up = nn.Linear(d_c, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, past_kv=None, seq_idx=None):
        B, T, D = x.shape
        q_compressed = self.q_down(x)
        q_c = q_compressed[..., :self.d_c]
        kv_compressed = self.kv_down(x)
        kv_c = kv_compressed[..., :self.d_c]

        q = self.q_up(q_c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_up(kv_c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(kv_c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)
        T_q = q.size(2)
        T_kv = k.size(2)

        if seq_idx is None and T_q <= self.window_size:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            try:
                from torch.nn.attention.flex_attention import create_block_mask
                if seq_idx is not None:
                    def mask_mod(b, h, q_idx, kv_idx):
                        same = seq_idx[0, q_idx] == seq_idx[0, kv_idx]
                        causal = kv_idx <= q_idx
                        window = kv_idx >= (q_idx - self.window_size + 1)
                        return same & causal & window
                else:
                    def mask_mod(b, h, q_idx, kv_idx):
                        causal = kv_idx <= q_idx
                        window = kv_idx >= (q_idx - self.window_size + 1)
                        return causal & window
                block_mask = create_block_mask(
                    mask_mod, B=None, H=None, Q_LEN=T_q, KV_LEN=T_kv,
                    device=x.device, BLOCK_SIZE=128,
                )
                out = _COMPILED_FLEX_ATTENTION(q, k, v, block_mask=block_mask)
            except Exception:
                q_pos = torch.arange(T_kv - T_q, T_kv, device=x.device).unsqueeze(1)
                k_pos = torch.arange(T_kv, device=x.device).unsqueeze(0)
                mask = (k_pos <= q_pos) & (k_pos >= q_pos - self.window_size + 1)
                if seq_idx is not None:
                    mask = mask & (seq_idx[0].unsqueeze(1) == seq_idx[0].unsqueeze(0))
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(out), new_kv