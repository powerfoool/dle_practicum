# %% [markdown]
# # Проект: Многоцелевая модель для NER + event-CLS
# 
# Этот Jupyter-ноутбук - пошаговый шаблон для выполнения проекта по объединённой (multi-task) модели, решающей **NER (BIO, token-level)** и **CLS (document-level multihot событий/отношений)** на датасете NEREL.
# 
# Модель должна решать параллельно две задачи: классифицировать токены на BIO-теги и классифицировать текст на принадлежность к 30 классам-характеристикам текста.
# 
# Характеристики:
# ```
# ['WORKPLACE', 'ALTERNATIVE_NAME', 'WORKS_AS', 'PARTICIPANT_IN', 'POINT_IN_TIME', 'TAKES_PLACE_IN', 'HEADQUARTERED_IN', 'ORIGINS_FROM', 'LOCATED_IN', 'AGENT', 'AGE_IS', 'HAS_CAUSE', 'PRODUCES', 'AWARDED_WITH', 'PART_OF', 'IDEOLOGY_OF', 'MEMBER_OF', 'CONVICTED_OF', 'INANIMATE_INVOLVED', 'SUBEVENT_OF', 'SUBORDINATE_OF', 'KNOWS', 'MEDICAL_CONDITION', 'PARENT_OF', 'PLACE_RESIDES_IN', 'OWNER_OF', 'ABBREVIATION', 'FOUNDED_BY', 'ORGANIZES', 'PENALIZED_AS']
# ```
# 
# Внимание: вам нужно реализовать весь рабочий код - в ноутбуке реализована только загрузка датасета. Все остальные ячейки служат как инструкции / места для вашего кода.
# 

# %% [markdown]
# #### Структура ноутбука
# 
# 1. Подготовка окружения (пути, seed, imports)
# 2. EDA - загрузка данных, обзор, графики, выводы
# 3. Токенизация, выравнивание меток, DataLoader - реализовать `tokenize_and_align_labels`, Dataset/Collator
# 4. Модель (JointModel) и кастомный loss (uncertainty-weighting) - реализовать модельный класс и loss
# 5. Тренировка/валидация - training loop, оптимизатор, scheduler, логирование метрик
# 6. Инференс и анализ ошибок - реализовать inference pipeline и примеры
# 
# 

# %% [markdown]
# ##### 1. EDA
# 
# Цели:
# 
# - Прочитать 3 записи датасета.
# - Посчитать встречаемость каждого класса в cls_vec.
# - Построить графики: распределение длины текстов, число сущностей на документ.
# - Написать 2–3 коротких вывода в Markdown: что можно ожидать при моделировании (редкие типы, длинные документы и т. п.).
# 
# 
# 

# %%
features = ['WORKPLACE', 'ALTERNATIVE_NAME', 'WORKS_AS', 'PARTICIPANT_IN', 'POINT_IN_TIME',
    'TAKES_PLACE_IN', 'HEADQUARTERED_IN', 'ORIGINS_FROM', 'LOCATED_IN', 'AGENT', 
    'AGE_IS', 'HAS_CAUSE', 'PRODUCES', 'AWARDED_WITH', 'PART_OF', 'IDEOLOGY_OF', 
    'MEMBER_OF', 'CONVICTED_OF', 'INANIMATE_INVOLVED', 'SUBEVENT_OF', 'SUBORDINATE_OF', 
    'KNOWS', 'MEDICAL_CONDITION', 'PARENT_OF', 'PLACE_RESIDES_IN', 'OWNER_OF', 
    'ABBREVIATION', 'FOUNDED_BY', 'ORGANIZES', 'PENALIZED_AS']

# %% [markdown]
# Посмотрим на датасет

# %%
from datasets import load_dataset
dataset = load_dataset("danasone/nerel")
dataset

# %% [markdown]
# Датасет состоит из сплита train. Для обучение выделим из него val и test подвыборки.
# 
# Выведем 3 записи датасета

# %%
for i in range(3):
    print(f"---\nТЕКСТ {i+1}:\n")
    for k in dataset["train"].features.keys():
        print(f"{k}:")
        print(dataset["train"][i][k])
        print()

# %% [markdown]
# Посчитаем встречаемость характеристик в датасете

# %%
features_cnt = {k: 0 for k in features}

for row in dataset["train"]:
    for feature, flag in zip(features, row["cls_vec"]):
        if flag:
            features_cnt[feature] += 1

features_cnt

# %% [markdown]
# Оценим кол-во сущностей в датасете

# %%
from collections import Counter

Counter([item.split("-")[1] for row in dataset["train"] for item in row["tags"] if "B-" in item])

# %% [markdown]
# Построим график распределения длин текстов и числа сущностей на документ

# %%
from matplotlib import pyplot as plt

text_lengths = [len(row["text"]) for row in dataset["train"]]

# график распределения длин текстов
plt.hist(text_lengths, bins=100)
plt.title("Длины текстов в символах")
plt.show()

# %%
texts = [row["text"] for row in dataset['train']]

print("---\nТоп 3 самых коротких текста\n")
for idx, text in enumerate(sorted(texts, key=len)[:3]):
    print(f"Текст {idx+1}\n")
    print(text + "\n")

# %%
print("Топ 3 самых длинных текста\n")
for idx, text in enumerate(sorted(texts, key=len, reverse=True)[:3]):
    print(f"---\nТекст {idx+1}\n")
    print(text + "\n")

# %%
entity_cnt = []
for row in dataset["train"]:
    entity_cnt.append(len([item for item in row["tags"] if item.startswith("B-")]))

plt.hist(entity_cnt, bins=100)
plt.title("Кол-во сущностей на текст")
plt.show()

# %% [markdown]
# ### Выводы по EDA
# - и сущности и отношения (характеристики) которые нужно распознавать представлены в датасете неравномерно, распознавание редких сущности будет сложнее выучить
# - также будет сложно выучить распознавание сущностей в длинных текстах (в датасете есть выбросы с очень большим кол-вом символов), где модели нужно выстаивать связи большого кол-ва токенов друг с другом

# %% [markdown]
# ---

# %% [markdown]
# ##### 3. Токенизация и выравнивание меток
# 
# Задачи:
# 
# - Выбрать `AutoTokenizer(..., use_fast=True)`.
# - Реализовать `tokenize_and_align_labels(examples, tokenizer, label2id, max_length)`:
#   - Токенизировать текст (return_offsets_mapping
#   - Преобразовать word-level BIO метки в token-level метки (subword → label = -100 / ignore_index, для первых субтокенов ставится соответствующий тег `B-`/`I-`)
#   - Вернуть словарь с `input_ids`, `attention_mask`, `labels` (token-level), `cls_labels`
# 
# - Собрать `torch.utils.data.Dataset` и `DataLoader`. Можно использовать `DataCollatorForTokenClassification` либо сделать кастомный collator, который возвращает батчи с `cls_labels`.
# 
# 

# %% [markdown]
# ---
# Тексты в датасете на русском. Нужно использовать токенизатор, который поддерживает русский язык.
# 
# Посмотрим на длину текстов в токенах

# %%
from transformers import AutoTokenizer

model_name = 'markussagen/xlm-roberta-longformer-base-4096'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenized_texts = tokenizer(list(dataset["train"]["text"]), truncation=False)

token_lengths = [len(tokens) for tokens in tokenized_texts["input_ids"]]
plt.hist(token_lengths, bins=100)
plt.title("Длины текстов в токенах")
plt.show()

# %% [markdown]
# Много текстов довольно длинные, поэтому был выбран трансформер xlm-roberta-longformer-base-4096 - мультиязычный roBERTa трансформер обученный с длиной контекста до 4096 токенов.
# 
# Перенесем BIO-метки с уровня слов на уровень токенов

# %%
entity_labels = list(set([item.split("-")[1] for row in dataset["train"] for item in row["tags"] if "B-" in item]))
label2id = {lab: i for i, lab in enumerate(entity_labels)}
id2label = {i: lab for lab, i in label2id.items()}

def tokenize_and_align_labels(tokenizer, label2id, max_length=128):
    def _tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            list(examples["text"]),
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=max_length)
        labels = []
        for i, word_labels in enumerate(examples["tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    label_id = -100
                    word_entity = "O"
                    if len(word_labels[word_idx].split("-")) == 2:
                        word_entity = word_labels[word_idx].split("-")[1]
                    label_id = label2id.get(word_entity, -100)
                    label_ids.append(label_id)
                else:
                    label_ids.append(-100)
                prev_word_idx = word_idx
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized
    return _tokenize_and_align_labels

tokenized_dataset = dataset["train"].map(
    tokenize_and_align_labels(tokenizer, label2id, max_length=1024),
    batched=True,
    remove_columns=["text", "tokens", "tags", "token_spans"],
)

# %% [markdown]
# Проверим что перенос корректный

# %%
list(zip(
    tokenizer.convert_ids_to_tokens(tokenized_dataset["input_ids"][0]),
    tokenized_dataset["labels"][0],
    [id2label.get(tkn, "") for tkn in tokenized_dataset["labels"][0]]
))[:100]

# %% [markdown]
# Объявляем `DataLoader` со стандартным `collate_fn=DataCollatorForTokenClassification`

# %%
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader

data_collator = DataCollatorForTokenClassification(tokenizer)
train_dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)

# %% [markdown]
# ---

# %% [markdown]
# ##### 4. Модель: `JointModel` + custom loss (uncertainty weighting)
# 

# %% [markdown]
# У нас готовы все данные, давайте создадим модель, которую потом будем обучать. К базовой модели, которую вы взяли на этапе выбора токенизатора, добавьте слой Dropout, линейный слой классификации сущностей и линейный слой классификации характеристик. Обратите внимание, что характеристики связаны со всем текстом, поэтому надо из всех эмбеддингов выбрать один, например, с помощью `[CLS]` пулинга. Обратите внимание, что `[CLS]` в квадратных скобках обозначает специальный токен, вектор которого нужно использовать для классификации текста.
# 
# Также реализуйте лосс-функцию для совместного обучения. Самый простой —  сумма лоссов для каждой задачи loss = token_loss + cls_loss.

# %%
import torch
from torch import nn
from transformers import AutoModel
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()


class JointModel(nn.Module):
    def __init__(self, config, label2id, features, pos_weight=None):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(config.BASE_MODEL_NAME, ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout(p=config.DROPOUT)
        self.use_uncertainty_weight = config.USE_UNCERTAINTY_WEIGHT

        self.entity_cnt = len(label2id)
        self.features_cnt = len(features)
        self.cls_term_multiplier = config.CLS_TERM_MULTIPLIER

        self.token_fc = nn.Linear(self.base_model.config.hidden_size, self.entity_cnt)
        self.cls_fc = nn.Linear(self.base_model.config.hidden_size, self.features_cnt)

        self.log_sigma_token = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.cls_fc.weight)
        nn.init.zeros_(self.cls_fc.bias)

    def forward(self, input_ids, attention_mask, labels, cls_vec):
        embeddings = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        dropout_embeddings = self.dropout(embeddings.last_hidden_state)
        token_logits = self.token_fc(dropout_embeddings)
        # CLS-logits
        # cls_logits = self.cls_fc(dropout_embeddings[:, 0])

        # mean pooling
        mask = attention_mask.unsqueeze(-1)
        masked = dropout_embeddings * mask
        mean_pooled = masked.sum(1) / mask.sum(1).clamp(min=1)
        cls_logits = self.cls_fc(mean_pooled)

        token_loss = self.token_loss_fn(token_logits.view(-1, self.entity_cnt), labels.view(-1))
        cls_loss = self.cls_loss_fn(cls_logits, cls_vec.float())

        loss_token_term, loss_cls_term = torch.tensor(0.0), torch.tensor(0.0)
        if self.use_uncertainty_weight:
            loss_token_term = torch.exp(-2.0 * self.log_sigma_token) * token_loss + self.log_sigma_token
            loss_cls_term = torch.exp(-2.0 * self.log_sigma_cls) * cls_loss + self.log_sigma_cls
            loss = loss_token_term + loss_cls_term * self.cls_term_multiplier
        else:
            loss = token_loss + cls_loss * self.cls_term_multiplier

        return {
            "token_logits": token_logits,
            "cls_logits": cls_logits,
            "loss": loss,
            "loss_token_term": loss_token_term,
            "loss_cls_term": loss_cls_term,
        }

# %% [markdown]
# ##### 5. Training / Validation
# 
# 

# %% [markdown]
# Когда логика модели прописана, можем начать ее обучать. Вот основые шаги, которые нужно выполнить:
# - Настройте оптимизатор, LR-scheduler, gradient clipping.
# - Обучите на достаточном количестве эпох. Можете воспользоваться значением с предыдущих уроков. Можно начать с 5 эпох и увеличивать, пока лосс не перестанет уменьшаться или пока не увидите переобучение.
# - Рассчитайте качество на тестовой выборке. Используйте f1_score из библиотеки sklearn для каждой задачи:
#    - По всем сущностям — F1 macro
#    - По всем классам — F1 micro
# - Соберите логи: для каждой эпохи выведите loss, token_f1, cls_f1.
# - Сделайте выводы по проведённому обучению. На какой эпохе она обучилась, насколько лосс на обучении отличается от лосса на валидации на каждой задачи.

# %%
import torch
import random
import numpy as np
from tqdm.auto import tqdm


class Config:
    SEED = 42
    BASE_MODEL_NAME = "markussagen/xlm-roberta-longformer-base-4096"

    # Гиперпараметры
    BATCH_SIZE = 16
    BASE_LEARNING_RATE = 2e-5
    TOKEN_LEARNING_RATE = 5e-4
    CLS_LEARNING_RATE = 5e-4
    UNCERTAINTY_LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 5
    DROPOUT = 0.1
    MAX_TOKENS = 1024
    UNFREEZE_LAYERS_PATTERN = "pooler|encoder\.layer\.(8|9|10|11)"
    USE_UNCERTAINTY_WEIGHT = True
    CLS_TERM_MULTIPLIER = 10.

    MODELS_PATH = "models"


def set_seed(seed=42):
    """
    Устанавливает сиды для всех источников случайности в PyTorch.
    """
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Python
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Для некоторых операций — принудительно детерминистические алгоритмы
    torch.use_deterministic_algorithms(True, warn_only=True)

config = Config()
set_seed(config.SEED)

# %% [markdown]
# Изучим базовую модель, чтобы понять как к ней подступиться

# %%
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

print(AutoModel.from_pretrained(config.BASE_MODEL_NAME, ignore_mismatched_sizes=True))

# %% [markdown]
# Получается слой эмбедингов, 12 энкодеров и pooler слой для работы с представление CLS-токена. Для NER на нужен выход последнего энкодера, а для второй задачи мультиклассовой классификации нужен выход pooler-слоя

# %%
import torch
from torch.utils.data import random_split
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import Subset

tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, use_fast=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

entity_labels = list(set([item.split("-")[1] for row in dataset["train"] for item in row["tags"] if "B-" in item]))
label2id = {lab: i for i, lab in enumerate(entity_labels)}

tokenized_dataset = dataset["train"].map(
    tokenize_and_align_labels(tokenizer, label2id, max_length=config.MAX_TOKENS),
    batched=True,
    remove_columns=["text", "tokens", "tags", "token_spans"],
)

# tokenized_dataset = Subset(tokenized_dataset, list(range(50)))

train_dataset, val_dataset = random_split(tokenized_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(config.SEED))

train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=data_collator)

# %%
import numpy as np
from collections import Counter

# Собираем все векторы меток из train_dataset
all_labels = []
for sample in train_dataset:
    # cls_vec – это список/тензор из 0 и 1 длиной features_cnt
    all_labels.append(sample["cls_vec"])

# Преобразуем в numpy массив (N_samples, features_cnt)
all_labels = np.array(all_labels)

# Для каждого класса считаем pos_weight = num_negatives / num_positives
# Чтобы избежать деления на ноль, если положительных примеров нет, ставим большой вес или 1.0
num_pos = np.sum(all_labels, axis=0)
num_neg = all_labels.shape[0] - num_pos

# Стандартная формула pos_weight = num_neg / num_pos
pos_weight = np.where(num_pos > 0, num_neg / num_pos, 1.0)  # если 0 положительных, weight=1

print("pos_weight per class:", pos_weight)

# %%
from torchinfo import summary
import re


def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for name, param in module.named_parameters():
            param.requires_grad = False
        return
    pattern = re.compile(unfreeze_pattern)
    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


model = JointModel(config, label2id, features, pos_weight)
set_requires_grad(model.base_model, unfreeze_pattern=config.UNFREEZE_LAYERS_PATTERN, verbose=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
print(f"Модель перенесена на девайс: {device}")
if torch.cuda.is_available():
    print(f"Девайс: {torch.cuda.get_device_name(0)}")

optimizer = torch.optim.AdamW([
        {'params': model.base_model.parameters(), 'lr': config.BASE_LEARNING_RATE, 'name': 'base_model'},
        {'params': model.token_fc.parameters(), 'lr': config.TOKEN_LEARNING_RATE, 'name': 'token_fc'},
        {'params': model.cls_fc.parameters(), 'lr': config.CLS_LEARNING_RATE, 'name': 'cls_fc'},
        {"params": [model.log_sigma_token, model.log_sigma_cls], "lr": config.UNCERTAINTY_LEARNING_RATE, 'name': 'uncertainty'},
    ], weight_decay=config.WEIGHT_DECAY
)

print("Сводка о модели")
summary(model)

# %%
from sklearn.metrics import f1_score
import json
import os
from datetime import datetime
from hashlib import sha256


def train_model(config, train_dataloader, val_dataloader, train_history=None):
    epoch_to_start = 1
    best_loss = 100
    if train_history is None:
        train_history = []
    if len(train_history) > 0:
        epoch_to_start = train_history[-1]["epoch"] + 1
        best_loss = train_history[-1]["val_loss"]

    scheduler = init_scheduler()

    for epoch in range(epoch_to_start, config.EPOCHS + epoch_to_start):
        model.train()
        device = next(model.parameters()).device

        epoch_loss, loss_token_term, loss_cls_term = 0.0, 0., 0.
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            if 'offset_mapping' in inputs:
                del inputs['offset_mapping']
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            loss_token_term += outputs["loss_token_term"].item()
            loss_cls_term += outputs["loss_cls_term"].item()

        train_loss = epoch_loss / len(train_dataloader)
        train_token_loss = loss_token_term / len(train_dataloader)
        loss_cls_loss = loss_cls_term / len(train_dataloader)

        val_losses = validate_model(val_dataloader)
        current_lrs = {group['name']: group['lr'] for group in optimizer.param_groups}

        train_history_metadatum = {
            "epoch": epoch,
            "learning_rates": current_lrs,
            "train_loss": train_loss,
            "val_loss": val_losses["loss"],
            "val_token_f1": val_losses["token_f1"],
            "val_cls_f1": val_losses["cls_f1"]
        }
        train_history.append(train_history_metadatum)
        if (val_losses["cls_f1"] >= 0.75
                and val_losses["token_f1"] >= 0.5
                and val_losses["loss"] < best_loss):
            best_loss = val_losses["loss"]
            save_model(model, config,
                additional_info=dict(train_history=train_history, **train_history_metadatum))

        print(
            f"train loss: {train_loss:.4f}\n"
            f"train token loss: {train_token_loss:.4f}\n"
            f"train cls loss: {loss_cls_loss:.4f}\n"
            f"val loss: {val_losses['loss']:.4f}\n"
            f"val token F1: {val_losses['token_f1']:.4f}\n"
            f"val cls F1: {val_losses['cls_f1']:.4f}"
        )
    return train_history


def init_scheduler():
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[
            config.BASE_LEARNING_RATE,
            config.TOKEN_LEARNING_RATE,
            config.CLS_LEARNING_RATE,
            config.UNCERTAINTY_LEARNING_RATE,
        ],
        steps_per_epoch=len(train_dataloader),
        epochs=config.EPOCHS,
        pct_start=0.1,            # 10% времени – подъём
        anneal_strategy='cos',    # косинусный спад в конце
        div_factor=25.0,          # начальный lr = max_lr / 25
        final_div_factor=1000.0,  # конечный lr = max_lr / 1000
    )
    return scheduler


def validate_model(val_dataloader):
    model.eval()
    device = next(model.parameters()).device
    epoch_val_loss = 0.0
    token_labels, token_predicts, cls_labels, cls_predicts = [], [], [], []
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            if 'offset_mapping' in inputs:
                del inputs['offset_mapping']
            outputs = model(**inputs)
            loss = outputs["loss"]
            epoch_val_loss += loss.item()

            token_preds = torch.argmax(outputs["token_logits"], dim=-1).view(-1).cpu().tolist()
            labels = inputs["labels"].view(-1).cpu().tolist()
            for p, t in zip(token_preds, labels):
                if t != -100:
                    token_labels.append(int(t))
                    token_predicts.append(int(p))

            cls_labels.extend(inputs["cls_vec"].cpu().numpy())
            probs = torch.sigmoid(outputs["cls_logits"])
            cls_predicts.extend((probs > 0.5).int().cpu().numpy())

    token_f1 = f1_score(token_labels, token_predicts, average="macro", zero_division=0)
    cls_f1 = f1_score(cls_labels, cls_predicts, average="micro", zero_division=0)

    return {
        "loss": epoch_val_loss / len(val_dataloader),
        "token_f1": token_f1,
        "cls_f1": cls_f1
    }


def serialize_config(config):
    attrs_source = dict(config.__class__.__dict__)
    attrs = {k: v for k, v in attrs_source.items() if not k.startswith("__") and not callable(v)}
    return json.dumps(attrs, indent=4)


def load_config(json_data):
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data 
    config_instance = config()
    for key, value in data.items():
        setattr(config_instance, key, value)
    return config_instance


def save_model(model, config, additional_info={}):
    config_serialized = serialize_config(config)
    current_datetime = datetime.now()
    model_id = sha256(f"{config_serialized}/{current_datetime}".encode()).hexdigest()
    
    metadata = dict()
    metadata_path = f"{config.MODELS_PATH}/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r+") as file:
            metadata = json.load(file)
    metadata[model_id] = {
        "config": config_serialized,
        **additional_info,
    }
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)
    
    model_path = f"{config.MODELS_PATH}/{model_id}.pth"
    torch.save(model.state_dict(), model_path)

    return model_path

# %%
train_history = train_model(config, train_dataloader, val_dataloader)

# %%
train_history = train_model(config, train_dataloader, val_dataloader, train_history)

# %%
train_history = train_model(config, train_dataloader, val_dataloader, train_history)

# %%
train_history = train_model(config, train_dataloader, val_dataloader, train_history)

# %%
# train_history = train_model(config, train_dataloader, val_dataloader, train_history)

# %%
# train_history = train_model(config, train_dataloader, val_dataloader, train_history)

# %%
import json

with open("models/metadata.json", "r") as file:
    metadata = json.load(file)

best_loss = 100
for model_id, metadatum in metadata.items():
    if metadatum["val_loss"] < best_loss and metadatum["val_loss"] > 1.:
        best_loss = metadatum["val_loss"]
        best_model_id = model_id

best_train_history = metadata[best_model_id]["train_history"]

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

epochs = [entry['epoch'] for entry in best_train_history]
train_loss = [entry['train_loss'] for entry in best_train_history]
val_loss = [entry['val_loss'] for entry in best_train_history]
val_token_f1 = [entry['val_token_f1'] for entry in best_train_history]
val_cls_f1 = [entry['val_cls_f1'] for entry in best_train_history]

all_lr_types = set()
for entry in best_train_history:
    if 'learning_rates' in entry:
        all_lr_types.update(entry['learning_rates'].keys())
lr_data = {lr_type: [] for lr_type in all_lr_types}

for entry in best_train_history:
    if 'learning_rates' in entry:
        lr_dict = entry['learning_rates']
        for lr_type in all_lr_types:
            lr_data[lr_type].append(lr_dict.get(lr_type, None))

# График 1: Train и Validation Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
plt.xlabel('Эпоха')
plt.title('Train и Validation Loss')
plt.legend()
plt.grid(True)

# График 2: Validation Token F1
plt.subplot(2, 2, 2)
plt.plot(epochs, val_token_f1, label='Val Token F1', color='green', marker='^')
plt.xlabel('Эпоха')
plt.title('Validation Token F1')
plt.legend()
plt.grid(True)

# График 3: Validation CLS F1
plt.subplot(2, 2, 3)
plt.plot(epochs, val_cls_f1, label='Val CLS F1', color='orange', marker='D')
plt.xlabel('Эпоха')
plt.title('Validation CLS F1')
plt.legend()
plt.grid(True)

# График 4: Learning Rate
plt.subplot(2, 2, 4)

for idx, lr_type in enumerate(lr_data.keys()):
    data = lr_data[lr_type]
    plt.plot(epochs, data, label=f'{lr_type} LR', marker='o', markersize=4)

plt.xlabel('Эпоха')
plt.title('Learning Rates (все компоненты)')
plt.legend()
plt.grid(True)
plt.yscale('log')

# Финальная настройка
plt.tight_layout()
plt.show()

# %%


# %%


# %% [markdown]
# ##### 6. Инференс, квантизация и анализ ошибок
# 
# Проведите качественный анализ на 8–10 примерах: где NER ошибается? Какие типы сущностей плохо определяются? Насколько квантизация может ускорить инференс и сильно ли она ухудшает модель?
# 

# %% [markdown]
# ##### Заключение
# 
# Этот шаблон даёт вам чёткую дорожную карту и рабочие точки, где нужно реализовать код. В ноутбуке предоставлены только парсеры строкового формата - всё остальное вы пишете самостоятельно: токенизация/выравнивание меток, датасеты, модель, loss, тренировка и анализ.
# 
# Удачи - приступайте к реализации прямо в ноутбуке!

# %%


# %%


# %%


# %%


# %%


# %%
import os
import json


with open("models/metadata.json", "r") as file:
    metadata = json.load(file)

bad_models = []
for k, v in metadata.items():
    if v["val_loss"] > 0.5:
       bad_models.append(k)

for bad_model in bad_models:
    metadata.pop(bad_model)
    if os.path.exists(f"models/{bad_model}.pth"):
        os.remove(f"models/{bad_model}.pth")

with open("models/metadata.json", "w") as file:
    json.dump(metadata, file)

# %%



