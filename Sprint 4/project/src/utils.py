import os
import re
import json
import math
import random
import stat
import pandas as pd

import numpy as np
import paramiko
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from src.config import serialize_config
from src.dataset import MultimodalDataset


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


def train(model, train_loader, val_loader, config, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        device_name = ""
        if torch.cuda.is_available():
            device_name = f": \"{torch.cuda.get_device_name()}\""
        print(f"Обучение происходит на девайсе \"{device}\"{device_name}")

    model = model.to(device)

    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.regressor.parameters(), 'lr': config.REGRESSOR_LR},
    ])
    
    criterion = nn.L1Loss(reduction='sum')
    
    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=verbose)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=verbose)
    
    best_mae = 10e5
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss, count = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            total_calories = batch["total_calories"].to(device)
            total_mass = batch["total_mass"].to(device)
            targets = batch["calories_per_gram"].to(device)

            optimizer.zero_grad()
            predicts = model(**inputs)
            train_loss = criterion(predicts, targets)
            train_loss.backward()
            optimizer.step()

            total_loss += criterion(predicts * total_mass, total_calories).item()
            count += total_calories.shape[0]

        val_mae = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{config.EPOCHS} | train MAE: {total_loss/count:.4f} | val MAE: {val_mae:.4f}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            save_model(model, config, replace_best_model=True,
                additional_info={
                    "epoch": epoch+1,
                    "train_mae": total_loss/count,
                    "val_mae": val_mae,
                })


def batch2df(batch):
    df = {}
    for k, v in batch.items():
        v = v.tolist() if type(v) is torch.Tensor else v
        df[k] = v
    return pd.DataFrame(df)


def apply_model(model, val_loader, device):
    output = {k: [] for k in next(iter(val_loader)).keys()}
    output["predict"] = []
    output = pd.DataFrame(output)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            total_mass = batch["total_mass"].to(device)
            predicts = model(**inputs)

            batch["predict"] = predicts * total_mass
            output = pd.concat([output, batch2df(batch)], ignore_index=True)
    return output


def validate(model, val_loader, device):
    criterion = nn.L1Loss(reduction='sum')
    model.eval()
    loss, count = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            total_calories = batch["total_calories"].to(device)
            total_mass = batch["total_mass"].to(device)

            predicts = model(**inputs)
            loss += criterion(predicts * total_mass, total_calories).item()
            count += total_calories.shape[0]
    return loss / count


def save_model(model, config, additional_info={}, replace_best_model=False):
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

    if replace_best_model:
        best_model_path = f"{config.MODELS_PATH}/best_model.pth"
        try:
            os.symlink(model_path, best_model_path)
        except FileExistsError:
            os.remove(best_model_path)
            os.symlink(model_path, best_model_path)

    return model_path


def convert_size(size_bytes):
    """Конвертирует размер файла из байт в удобочитаемый формат"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def copy_new_files_sftp(
    hostname,
    username,
    password=None,
    private_key_path=None,
    remote_dir='/path/to/remote/directory',
    local_dir='/path/to/local/directory',
    force_copy=None,
):
    """
    Копирует файлы с удалённой машины, только если их ещё нет в локальной директории.
    """
    # Создаём локальную директорию, если её нет
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # Настройка SSH-клиента
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Подключение (с паролем или приватным ключом)
        if private_key_path:
            ssh_client.connect(
                hostname=hostname,
                username=username,
                key_filename=private_key_path,
                port=22,
                timeout=30,
                banner_timeout=60,
                auth_timeout=30,
            )
        else:
            ssh_client.connect(
                hostname=hostname,
                username=username,
                password=password
            )

        # Создание SFTP-сессии
        sftp = ssh_client.open_sftp()

        # Получение списка файлов на удалённой машине
        remote_files = sftp.listdir(remote_dir)

        copied_count = 0
        for filename in tqdm(remote_files):
            remote_path = f"{remote_dir}/{filename}"
            file_attr = sftp.lstat(remote_path)
            if stat.S_ISLNK(file_attr.st_mode):
                print(f"Пропущен симлинк: {filename}")
                continue

            file_attr = sftp.stat(remote_path)
            file_size_bytes = file_attr.st_size
            # Конвертируем в удобочитаемый формат
            file_size_readable = convert_size(file_size_bytes)

            local_path = os.path.join(local_dir, filename)

            if force_copy is None:
                force_copy = []
            # Проверяем, существует ли файл локально
            if not os.path.exists(local_path) or filename in force_copy:
                # Копируем файл
                sftp.get(remote_path, local_path)
                print(f"Скопирован: {filename} (размер: {file_size_readable})")
                copied_count += 1
            # else:
            #     print(f"Пропущен (уже существует): {filename}")

        print(f"\nВсего скопировано файлов: {copied_count}")

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        ssh_client.close()

