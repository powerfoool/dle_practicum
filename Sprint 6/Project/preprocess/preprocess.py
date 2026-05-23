#!/usr/bin/env python3
"""
Скрипт для препроцессинга датасета русской литературы.
Выполняет:
1. Загрузку всех текстовых файлов из папки dataset
2. Очистку от дубликатов
3. Фильтрацию предложений с некириллическими символами
4. Обработку повторяющейся пунктуации
5. Разбивку на чанки с добавлением <bos> и <eos> токенов
6. Обучение BPE токенизатора с размером словаря ~3000 токенов
"""

import os
import re
import glob
import json
from typing import List, Tuple, Set
from collections import defaultdict
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import numpy as np


def load_text_files(data_dir: str) -> List[str]:
    """
    Загружает все текстовые файлы из указанной директории.
    Возвращает список строк (каждая строка - содержимое файла).
    """
    texts = []
    file_pattern = os.path.join(data_dir, "*.txt")
    for file_path in glob.glob(file_pattern):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)
                print(f"Загружен файл: {os.path.basename(file_path)} ({len(text)} символов)")
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")
    return texts


def split_into_sentences(text: str) -> List[str]:
    """
    Разбивает текст на предложения по точкам, восклицательным и вопросительным знакам.
    Учитывает многоточие и сокращения (например, "т.д.").
    Упрощённый подход.
    """
    # Заменяем многоточие на специальный маркер, чтобы не разбивать по нему
    text = re.sub(r'\.\.\.', '…', text)
    # Заменяем сокращения с точками
    abbreviations = [r'т\.д\.', r'т\.п\.', r'др\.', r'пр\.', r'г\.', r'см\.', r'ст\.', r'кн\.', r'ч\.', r'с\.']
    for abbr in abbreviations:
        text = re.sub(abbr, abbr.replace('.', '@'), text)
    
    # Разбиваем по . ! ? … (многоточие)
    sentences = re.split(r'(?<=[.!?…]) +', text)
    
    # Восстанавливаем сокращения
    restored = []
    for sent in sentences:
        sent = sent.replace('@', '.')
        restored.append(sent.strip())
    
    # Убираем пустые предложения
    restored = [s for s in restored if s]
    return restored


def filter_cyrillic_sentences(sentences: List[str]) -> List[str]:
    """
    Оставляет только предложения, состоящие преимущественно из кириллических символов,
    пробелов, знаков пунктуации и цифр.
    """
    # Регулярное выражение для кириллических символов, пробелов, пунктуации и цифр
    cyrillic_pattern = re.compile(r'^[а-яёА-ЯЁ0-9\s\.,!?;:"\'\-–—()…]+$')
    filtered = []
    for sent in sentences:
        if cyrillic_pattern.match(sent):
            filtered.append(sent)
        else:
            # Можно также проверить процент кириллических символов
            # но для простоты используем строгое соответствие
            pass
    return filtered


def deduplicate_sentences(sentences: List[str]) -> List[str]:
    """
    Удаляет дубликаты предложений (точное совпадение).
    """
    seen = set()
    unique = []
    for sent in sentences:
        if sent not in seen:
            seen.add(sent)
            unique.append(sent)
    return unique


def clean_punctuation(text: str) -> str:
    """
    Очищает повторяющуюся пунктуацию (например, "!!!", "??", ",,").
    Заменяет множественные пробелы на один, удаляет неразрывные пробелы.
    """
    # Убираем повторяющиеся знаки препинания (оставляем один)
    text = re.sub(r'([!?])\1+', r'\1', text)  # !! -> !
    text = re.sub(r'(,)\1+', r'\1', text)     # ,, -> ,
    text = re.sub(r'(\.)\1+', r'\1', text)    # .. -> . (хотя .. обычно не встречается)
    # Убираем повторяющиеся дефисы, тире
    text = re.sub(r'(-)\1+', r'\1', text)
    text = re.sub(r'(—)\1+', r'\1', text)
    # Заменяем все whitespace символы (включая неразрывные пробелы, табуляции) на обычный пробел
    text = re.sub(r'\s+', ' ', text)
    # Удаляем пробелы перед знаками препинания (кроме открывающих скобок)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Удаляем пробелы после открывающих скобок и перед закрывающими
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    return text.strip()


def chunk_text(sentences: List[str], max_chunk_size: int = 512) -> List[str]:
    """
    Объединяет предложения в чанки примерно max_chunk_size символов.
    Добавляет специальные токены <bos> и <eos> в начале и конце каждого чанка.
    """
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in sentences:
        sent_len = len(sent)
        if current_length + sent_len + 1 <= max_chunk_size:  # +1 для пробела
            current_chunk.append(sent)
            current_length += sent_len + 1
        else:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_text = f"<bos> {chunk_text} <eos>"
                chunks.append(chunk_text)
            # Начинаем новый чанк с текущим предложением
            current_chunk = [sent]
            current_length = sent_len
    
    # Добавляем последний чанк
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunk_text = f"<bos> {chunk_text} <eos>"
        chunks.append(chunk_text)
    
    return chunks


def train_bpe_tokenizer(texts: List[str], vocab_size: int = 3000, save_path: str = "tokenizer.json"):
    """
    Обучает BPE токенизатор на предоставленных текстах.
    Сохраняет токенизатор в файл.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    
    # Инициализируем токенизатор с BPE моделью
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Создаём тренера с указанным размером словаря
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"],
        min_frequency=2
    )
    
    # Обучаем на текстах (передаём список строк)
    tokenizer.train_from_iterator(texts, trainer)
    
    # Сохраняем токенизатор
    tokenizer.save(save_path)
    print(f"Токенизатор сохранён в {save_path}")
    
    # Возвращаем токенизатор для дальнейшего использования
    return tokenizer


def main():
    data_dir = "dataset"
    output_dir = "processed"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Загрузка текстов ===")
    texts = load_text_files(data_dir)
    if not texts:
        print("Не найдено текстовых файлов. Проверьте путь.")
        return
    
    print(f"Загружено {len(texts)} файлов.")
    
    # Объединяем все тексты в одну строку для обработки
    full_text = "\n".join(texts)
    
    print("=== Разбивка на предложения ===")
    sentences = split_into_sentences(full_text)
    print(f"Получено {len(sentences)} предложений.")
    
    print("=== Фильтрация по кириллице ===")
    sentences = filter_cyrillic_sentences(sentences)
    print(f"После фильтрации осталось {len(sentences)} предложений.")
    
    print("=== Очистка от дубликатов ===")
    sentences = deduplicate_sentences(sentences)
    print(f"После удаления дубликатов осталось {len(sentences)} предложений.")
    
    print("=== Очистка пунктуации ===")
    cleaned_sentences = [clean_punctuation(s) for s in sentences]
    
    print("=== Разбивка на чанки ===")
    chunks = chunk_text(cleaned_sentences, max_chunk_size=512)
    print(f"Создано {len(chunks)} чанков.")
    
    # Сохраняем чанки в файл
    chunks_file = os.path.join(output_dir, "chunks.txt")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    print(f"Чанки сохранены в {chunks_file}")
    
    # Сохраняем предложения (без чанков) для возможного использования
    sentences_file = os.path.join(output_dir, "sentences.txt")
    with open(sentences_file, 'w', encoding='utf-8') as f:
        for sent in cleaned_sentences:
            f.write(sent + "\n")
    print(f"Предложения сохранены в {sentences_file}")
    
    print("=== Обучение BPE токенизатора ===")
    # Для обучения токенизатора используем предложения (без специальных токенов)
    # или чанки (с токенами). Обучим на предложениях, чтобы токенизатор учился на чистых словах.
    tokenizer = train_bpe_tokenizer(cleaned_sentences, vocab_size=3000, save_path=os.path.join(output_dir, "tokenizer.json"))
    
    # Демонстрация работы токенизатора
    test_sentence = "Это пример предложения для проверки токенизатора."
    encoded = tokenizer.encode(test_sentence)
    print(f"Тестовое предложение: {test_sentence}")
    print(f"Токены: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
    
    print("=== Препроцессинг завершён ===")


if __name__ == "__main__":
    main()