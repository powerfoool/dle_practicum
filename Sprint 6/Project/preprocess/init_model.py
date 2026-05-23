#!/usr/bin/env python3
"""
Скрипт для инициализации torch модели ~150M параметров с decoder-only архитектурой трансформера.
Используется конфигурация Llama с параметрами:
- hidden_size=1024
- intermediate_size=1536
- num_hidden_layers=16
- num_attention_heads=16
- num_key_value_heads=8
"""

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from typing import Optional
import json
import os


def estimate_parameters(config: LlamaConfig) -> int:
    """
    Оценивает количество параметров модели на основе конфигурации.
    Формула приблизительная, но даёт хорошую оценку.
    """
    # Параметры эмбеддингов (входные и выходные)
    vocab_params = config.vocab_size * config.hidden_size
    # Параметры слоёв трансформера
    # Каждый слой содержит:
    # - self_attn: q_proj, k_proj, v_proj, o_proj (каждый hidden_size x hidden_size)
    # - mlp: gate_proj, up_proj, down_proj (hidden_size x intermediate_size и обратно)
    # - layer_norm1, layer_norm2 (2 * hidden_size)
    per_layer = (
        4 * config.hidden_size * config.hidden_size  # q,k,v,o проекции
        + 2 * config.hidden_size * config.intermediate_size  # gate, up
        + config.intermediate_size * config.hidden_size  # down
        + 2 * config.hidden_size  # layer norms
    )
    total = vocab_params * 2 + per_layer * config.num_hidden_layers
    return total


def create_model(
    vocab_size: int = 3000,
    hidden_size: int = 1024,
    intermediate_size: int = 1536,
    num_hidden_layers: int = 16,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 8,
    max_position_embeddings: int = 2048,
    rope_theta: float = 10000.0,
    tie_word_embeddings: bool = False,
    device: Optional[str] = None,
) -> LlamaForCausalLM:
    """
    Создаёт модель Llama с заданными параметрами.
    
    Args:
        vocab_size: Размер словаря токенизатора.
        hidden_size: Размер скрытого состояния.
        intermediate_size: Размер промежуточного слоя MLP.
        num_hidden_layers: Количество слоёв трансформера.
        num_attention_heads: Количество голов внимания.
        num_key_value_heads: Количество голов для ключей/значений (GQA).
        max_position_embeddings: Максимальная длина последовательности.
        rope_theta: Параметр RoPE.
        tie_word_embeddings: Связывать ли входные и выходные эмбеддинги.
        device: Устройство для размещения модели ('cpu', 'cuda', 'mps').
    
    Returns:
        Модель LlamaForCausalLM.
    """
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        tie_word_embeddings=tie_word_embeddings,
        pad_token_id=0,  # <pad> токен
        bos_token_id=1,  # <bos> токен
        eos_token_id=2,  # <eos> токен
        use_cache=True,
    )
    
    model = LlamaForCausalLM(config)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model


def load_tokenizer_info(tokenizer_path: str = "tokenizer_3k.json") -> dict:
    """
    Загружает информацию о токенизаторе для определения vocab_size.
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Файл токенизатора не найден: {tokenizer_path}")
    
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    # Получаем размер словаря из токенизатора
    vocab_size = tokenizer_data.get("model", {}).get("vocab", {}).get("size", 3000)
    return {"vocab_size": vocab_size}


def main():
    """
    Основная функция: создаёт модель, оценивает параметры и сохраняет конфигурацию.
    """
    print("=== Инициализация модели трансформера ~150M параметров ===")
    
    # Пытаемся определить размер словаря из токенизатора
    tokenizer_path = "tokenizer_3k.json"
    vocab_size = 3000
    if os.path.exists(tokenizer_path):
        try:
            info = load_tokenizer_info(tokenizer_path)
            vocab_size = info["vocab_size"]
            print(f"Найден токенизатор, vocab_size = {vocab_size}")
        except Exception as e:
            print(f"Не удалось загрузить токенизатор: {e}, используем значение по умолчанию {vocab_size}")
    else:
        print(f"Токенизатор не найден по пути {tokenizer_path}, используем значение по умолчанию {vocab_size}")
    
    # Параметры модели (примерно 150M параметров)
    hidden_size = 1024
    intermediate_size = 1536
    num_hidden_layers = 16
    num_attention_heads = 16
    num_key_value_heads = 8
    
    print("Параметры конфигурации:")
    print(f"  vocab_size = {vocab_size}")
    print(f"  hidden_size = {hidden_size}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  num_hidden_layers = {num_hidden_layers}")
    print(f"  num_attention_heads = {num_attention_heads}")
    print(f"  num_key_value_heads = {num_key_value_heads}")
    
    # Создаём модель
    model = create_model(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )
    
    # Оцениваем количество параметров
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    estimated = estimate_parameters(config)
    
    print(f"\nОценка параметров:")
    print(f"  Реальное количество параметров: {total_params:,}")
    print(f"  Оценочное количество параметров: {estimated:,}")
    print(f"  Разница: {abs(total_params - estimated):,}")
    
    # Сохраняем конфигурацию модели
    config_path = "model_config.json"
    config_dict = config.to_dict()
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"\nКонфигурация модели сохранена в {config_path}")
    
    # Сохраняем модель (только веса, если нужно)
    model_path = "model_150m.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Веса модели сохранены в {model_path}")
    
    # Тестовый прогон
    print("\n=== Тестовый прогон ===")
    device = next(model.parameters()).device
    input_ids = torch.randint(0, vocab_size, (1, 16), device=device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        print(f"Входная форма: {input_ids.shape}")
        print(f"Выходная форма (logits): {logits.shape}")
        print(f"Устройство модели: {device}")
    
    print("\n=== Модель успешно создана ===")


if __name__ == "__main__":
    main()