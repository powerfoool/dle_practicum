import re


def text_clean(text):
    if not isinstance(text, str):
        return text
    # 1. К нижнему регистру
    text = text.lower()
    # 2. Удаляем ссылки, упоминания, эмодзи
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', '', text)
    # 3. Оставляем только буквы, цифры, пробелы; заменяем остальное на пробел
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', ' ', text)
    # 4. Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text



