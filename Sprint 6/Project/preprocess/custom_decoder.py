#!/usr/bin/env python3
"""
Простая кастомная реализация decoder-only трансформера для понимания архитектуры.
Не для production, а для образовательных целей.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Реализация Rotary Positional Embedding (RoPE) для self-attention.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Применяет RoPE к тензору x формы (batch, seq_len, heads, dim).
        """
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention с Grouped Query Attention (GQA) поддержкой.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Проекции для queries, keys, values
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Проекции
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Применяем RoPE
        cos, sin = self.rope(q, seq_len)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Транспонируем для attention
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Если количество голов ключей/значений меньше, чем голов запросов, повторяем
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        return output


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Применяет rotary positional embedding к тензору x.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated


class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) с SwiGLU активацией.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)


class DecoderLayer(nn.Module):
    """
    Один слой decoder-only трансформера.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, dropout)
        self.mlp = FeedForward(hidden_size, intermediate_size)
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention с residual connection
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x, mask)
        x = residual + self.dropout(attn_output)
        
        # MLP с residual connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x)
        x = residual + self.dropout(mlp_output)
        return x


class CustomDecoderOnlyTransformer(nn.Module):
    """
    Кастомная decoder-only модель трансформера.
    """
    def __init__(
        self,
        vocab_size: int = 3000,
        hidden_size: int = 1024,
        num_layers: int = 16,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        intermediate_size: int = 1536,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, num_kv_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Создаём causal mask
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)
        
        # Эмбеддинги токенов
        x = self.embed_tokens(input_ids)
        
        # Проход через слои
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_custom_model() -> CustomDecoderOnlyTransformer:
    """
    Создаёт кастомную модель с параметрами ~150M.
    """
    model = CustomDecoderOnlyTransformer(
        vocab_size=3000,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        num_kv_heads=8,
        intermediate_size=1536,
        max_seq_len=2048,
        dropout=0.1,
    )
    return model


if __name__ == "__main__":
    print("=== Кастомная decoder-only модель трансформера ===")
    model = create_custom_model()
    total_params = model.count_parameters()
    print(f"Количество параметров: {total_params:,}")
    
    # Тестовый прогон
    input_ids = torch.randint(0, 3000, (1, 32))
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Вход: {input_ids.shape}")
        print(f"Выход (logits): {logits.shape}")
    
    print("\nМодель успешно создана.")