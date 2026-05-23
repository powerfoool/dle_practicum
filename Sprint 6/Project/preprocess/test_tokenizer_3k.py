#!/usr/bin/env python3
"""
Тестирование BPE токенизатора с размером словаря 3000 токенов.
Обучается на всём датасете (или на большой выборке) и проверяется качество токенизации.
"""
import sys
sys.path.insert(0, '.')

from preprocess import (
    load_text_files,
    split_into_sentences,
    filter_cyrillic_sentences,
    deduplicate_sentences,
    clean_punctuation,
    train_bpe_tokenizer
)
import os
import json
from tokenizers import Tokenizer

def prepare_training_data():
    """Подготавливает данные для обучения токенизатора."""
    print("Загрузка файлов...")
    texts = load_text_files("dataset")
    full_text = "\n".join(texts)
    
    print("Разбивка на предложения...")
    sentences = split_into_sentences(full_text)
    print(f"Всего предложений: {len(sentences)}")
    
    print("Фильтрация по кириллице...")
    sentences = filter_cyrillic_sentences(sentences)
    print(f"После фильтрации: {len(sentences)}")
    
    print("Удаление дубликатов...")
    sentences = deduplicate_sentences(sentences)
    print(f"Уникальных предложений: {len(sentences)}")
    
    print("Очистка пунктуации...")
    sentences = [clean_punctuation(s) for s in sentences]
    
    # Ограничим количество предложений для ускорения (можно убрать)
    # sentences = sentences[:50000]
    
    return sentences

def test_tokenizer(tokenizer, test_sentences):
    """Тестирует токенизатор на наборе предложений."""
    print("\n=== Тестирование токенизатора ===")
    for i, sent in enumerate(test_sentences[:5]):
        encoded = tokenizer.encode(sent)
        print(f"\nПредложение {i+1}: {sent[:80]}...")
        print(f"   Токены: {encoded.tokens[:20]}{'...' if len(encoded.tokens) > 20 else ''}")
        print(f"   Количество токенов: {len(encoded.tokens)}")
        print(f"   IDs: {encoded.ids[:10]}{'...' if len(encoded.ids) > 10 else ''}")
    
    # Статистика
    total_tokens = 0
    total_chars = 0
    for sent in test_sentences[:100]:
        encoded = tokenizer.encode(sent)
        total_tokens += len(encoded.tokens)
        total_chars += len(sent)
    if total_chars > 0:
        avg_ratio = total_tokens / total_chars
        print(f"\nСреднее отношение токенов/символов (на 100 предложениях): {avg_ratio:.3f}")
    
    # Проверка специальных токенов
    special_tokens = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>"]
    print("\nПроверка специальных токенов:")
    for tok in special_tokens:
        try:
            id_ = tokenizer.token_to_id(tok)
            print(f"  {tok}: ID = {id_}")
        except:
            print(f"  {tok}: не найден в словаре")

def main():
    # Подготовка данных
    sentences = prepare_training_data()
    if not sentences:
        print("Нет данных для обучения.")
        return
    
    # Обучаем токенизатор с vocab_size=3000
    vocab_size = 3000
    save_path = "tokenizer_3k.json"
    print(f"\nОбучение BPE токенизатора с размером словаря {vocab_size}...")
    tokenizer = train_bpe_tokenizer(sentences, vocab_size=vocab_size, save_path=save_path)
    
    # Загружаем обратно для проверки
    tokenizer_loaded = Tokenizer.from_file(save_path)
    
    # Тестовые предложения
    test_sentences = [
        "Это пример предложения для проверки токенизатора.",
        "Ветер дул с севера, и небо покрылось тучами.",
        "Александр Сергеевич Пушкин — великий русский поэт.",
        "Они шли по улице, разговаривая о будущем.",
        "Стояла глубокая осень, листья уже пожелтели.",
        "Не сохами-то славная землюшка наша распахана…",
        "Распахана наша землюшка лошадиными копытами.",
        "Воротца со скотиньего база ведут на север к Дону."
    ]
    
    # Тестируем
    test_tokenizer(tokenizer_loaded, test_sentences)
    
    # Дополнительно: посмотрим на размер словаря
    vocab = tokenizer_loaded.get_vocab()
    print(f"\nРазмер словаря: {len(vocab)}")
    
    # Примеры частых токенов
    print("\nПримеры токенов (первые 20):")
    for i, (token, idx) in enumerate(list(vocab.items())[:20]):
        print(f"  {idx:4d}: '{token}'")
    
    # Сохраним информацию о токенизаторе
    info = {
        "vocab_size": len(vocab),
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"],
        "training_sentences": len(sentences),
        "save_path": save_path
    }
    with open("tokenizer_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"\nИнформация сохранена в tokenizer_info.json")

if __name__ == "__main__":
    main()