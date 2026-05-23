#!/usr/bin/env python3
"""
Обучение модели с использованием SFTTrainer из trl.
Включает коллбэки для валидации качества на промптах.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_from_disk
import numpy as np
from typing import Dict, List, Optional
import json
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Тестовые промпты для оценки качества
TEST_PROMPTS = [
    "Все мысли, которые имеют огромные последствия",
    "Сила войска зависит от его духа",
    "Мысль о том, что он принес страдания",
    "Человек сознает себя свободным",
    "Что бы ни случилось, я всегда буду",
    "Любовь мешает смерти",
    "Нет, жизнь не кончена",
    "Всякая мысль, даже самая простая",
    "Война не любезность, а самое гадкое дело",
    "Чтобы жить честно"
]

class PromptEvalCallback:
    """
    Коллбэк для оценки качества генерации на тестовых промптах.
    Вызывается в конце каждой эпохи.
    """
    def __init__(self, tokenizer, model, prompts=TEST_PROMPTS, max_length=50, device=None):
        self.tokenizer = tokenizer
        self.model = model
        self.prompts = prompts
        self.max_length = max_length
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Генерирует тексты для каждого промпта и выводит их.
        """
        logger.info(f"\n=== Оценка качества после эпохи {epoch} ===")
        generated_texts = []
        
        with torch.no_grad():
            for prompt in self.prompts:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                # Генерация с типичными параметрами
                output_ids = self.model.generate(
                    input_ids,
                    max_length=self.max_length,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_texts.append((prompt, generated))
                logger.info(f"Промпт: {prompt}")
                logger.info(f"Сгенерировано: {generated}")
                logger.info("---")
        
        # Сохраняем результаты в файл
        output_file = f"generation_epoch_{epoch}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for prompt, generated in generated_texts:
                f.write(f"Промпт: {prompt}\n")
                f.write(f"Сгенерировано: {generated}\n\n")
        logger.info(f"Результаты сохранены в {output_file}")
        
        # Возвращаем модель в режим обучения
        self.model.train()


def load_tokenizer(tokenizer_path: str = "tokenizer.json"):
    """
    Загружает токенизатор из файла.
    """
    from tokenizers import Tokenizer
    tokenizer_obj = Tokenizer.from_file(tokenizer_path)
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_model(config_path: Optional[str] = None, vocab_size: int = 3000) -> LlamaForCausalLM:
    """
    Создаёт модель Llama с конфигурацией из файла или по умолчанию.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = LlamaConfig.from_dict(config_dict)
    else:
        # Конфигурация по умолчанию ~150M параметров
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=1024,
            intermediate_size=1536,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=8,
            max_position_embeddings=2048,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
        )
    
    model = LlamaForCausalLM(config)
    logger.info(f"Модель создана, параметров: {sum(p.numel() for p in model.parameters()):,}")
    return model


def load_dataset(data_path: str = "dataset_tokenized") -> Dataset:
    """
    Загружает предобработанный датасет.
    """
    if os.path.exists(data_path):
        logger.info(f"Загрузка датасета из {data_path}")
        dataset = load_from_disk(data_path)
    else:
        # Если датасета нет, создадим фиктивный для демонстрации
        logger.warning(f"Датасет {data_path} не найден, создаём фиктивный.")
        # В реальности нужно использовать предобработанные данные
        # Создадим фиктивный датасет с 1000 примеров
        data = {
            "input_ids": [list(range(0, 128)) for _ in range(1000)],
            "attention_mask": [[1] * 128 for _ in range(1000)]
        }
        dataset = Dataset.from_dict(data)
    return dataset


def format_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """
    Форматирует датасет для SFT: добавляет промпты и ответы.
    В данном случае у нас уже есть input_ids, поэтому просто обрезаем до max_length.
    """
    def tokenize_function(examples):
        # examples уже содержат input_ids и attention_mask
        # Обрезаем до max_length
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        for ids, mask in zip(examples["input_ids"], examples["attention_mask"]):
            if len(ids) > max_length:
                ids = ids[:max_length]
                mask = mask[:max_length]
            # Для causal LM метки такие же как input_ids
            result["input_ids"].append(ids)
            result["attention_mask"].append(mask)
            result["labels"].append(ids)
        return result
    
    formatted = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return formatted


def compute_metrics(eval_pred):
    """
    Вычисляет метрики для оценки (perplexity).
    """
    logits, labels = eval_pred
    # Сдвигаем лейблы и логиты для вычисления потерь
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(torch.tensor(loss)).item()
    return {"perplexity": perplexity, "loss": loss.item()}


def main():
    # Параметры обучения
    batch_size = 96  # в диапазоне 64–128
    gradient_accumulation_steps = 2
    effective_batch_size = batch_size * gradient_accumulation_steps
    weight_decay = 0.01  # параметр регуляризации
    learning_rate = 5e-4
    num_train_epochs = 3
    logging_steps = 100
    save_steps = 500
    eval_steps = 500
    max_seq_length = 512
    
    logger.info("=== Загрузка токенизатора ===")
    tokenizer = load_tokenizer("tokenizer.json")
    vocab_size = tokenizer.vocab_size
    logger.info(f"Размер словаря: {vocab_size}")
    
    logger.info("=== Загрузка датасета ===")
    dataset = load_dataset("dataset_tokenized")
    logger.info(f"Датасет загружен, размер: {len(dataset)}")
    
    # Разделение на train/validation (если нет готового)
    if "validation" not in dataset:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    
    logger.info(f"Обучающая выборка: {len(train_dataset)}")
    logger.info(f"Валидационная выборка: {len(eval_dataset)}")
    
    # Форматируем датасет
    train_dataset = format_dataset(train_dataset, tokenizer, max_seq_length)
    eval_dataset = format_dataset(eval_dataset, tokenizer, max_seq_length)
    
    logger.info("=== Создание модели ===")
    model = create_model(vocab_size=vocab_size)
    
    # Коллбэк для оценки качества
    prompt_callback = PromptEvalCallback(tokenizer, model)
    
    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir="./trainer_output",
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )
    
    # Data collator для completion-only language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Инициализация SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        compute_metrics=compute_metrics,
        callbacks=[prompt_callback],
    )
    
    logger.info("=== Начало обучения ===")
    trainer.train()
    
    logger.info("=== Сохранение модели ===")
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    logger.info("Модель сохранена в ./final_model")


if __name__ == "__main__":
    main()