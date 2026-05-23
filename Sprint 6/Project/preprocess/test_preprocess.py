#!/usr/bin/env python3
"""
Тестовый запуск препроцессинга на 2 файлах.
"""
import sys
sys.path.insert(0, '.')

from preprocess import (
    load_text_files,
    split_into_sentences,
    filter_cyrillic_sentences,
    deduplicate_sentences,
    clean_punctuation,
    chunk_text,
    train_bpe_tokenizer
)
import os
import glob

def test_load():
    print("=== Тест загрузки файлов ===")
    # Ограничимся двумя файлами
    files = glob.glob("dataset/*.txt")[:2]
    texts = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            texts.append(fp.read())
        print(f"Загружен {os.path.basename(f)}")
    return texts

def main():
    texts = test_load()
    full_text = "\n".join(texts)
    
    print("=== Разбивка на предложения ===")
    sentences = split_into_sentences(full_text)
    print(f"Предложений: {len(sentences)}")
    if sentences:
        print("Пример предложения:", sentences[0][:100])
    
    print("=== Фильтрация по кириллице ===")
    filtered = filter_cyrillic_sentences(sentences)
    print(f"После фильтрации: {len(filtered)}")
    
    print("=== Очистка от дубликатов ===")
    unique = deduplicate_sentences(filtered)
    print(f"Уникальных: {len(unique)}")
    
    print("=== Очистка пунктуации ===")
    cleaned = [clean_punctuation(s) for s in unique[:10]]  # только первые 10
    print("Пример очищенного:", cleaned[0] if cleaned else "нет")
    
    print("=== Разбивка на чанки (тест) ===")
    chunks = chunk_text(unique[:50], max_chunk_size=200)  # маленький размер для теста
    print(f"Создано чанков: {len(chunks)}")
    if chunks:
        print("Пример чанка:", chunks[0][:150])
    
    print("=== Обучение BPE (упрощённое) ===")
    # Обучим на небольшом количестве предложений
    train_data = unique[:100]
    try:
        tokenizer = train_bpe_tokenizer(train_data, vocab_size=500, save_path="test_tokenizer.json")
        test_sentence = "Это тестовое предложение."
        encoded = tokenizer.encode(test_sentence)
        print(f"Тестовое предложение: {test_sentence}")
        print(f"Токены: {encoded.tokens}")
        print(f"IDs: {encoded.ids}")
    except Exception as e:
        print(f"Ошибка при обучении токенизатора: {e}")
    
    print("=== Тест завершён ===")

if __name__ == "__main__":
    main()