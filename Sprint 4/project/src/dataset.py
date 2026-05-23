import numpy as np
import pandas as pd
import random
import torch
import timm
import albumentations as A

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer


class MultimodalDataset(Dataset):
    def __init__(self, config, df, type="train"):
        super().__init__()
        self.images_path = config.IMAGES_PATH
        self.seed = config.SEED
        self.df = df
        self.type = type
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = self.get_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mega_dish_id = self.df.iloc[idx].mega_dish_id
        total_calories = self.df.iloc[idx].total_calories
        total_mass = self.df.iloc[idx].total_mass
        calories_per_gram = self.df.iloc[idx].calories_per_gram
        ingredient_names = self.df.iloc[idx].ingredient_names
        img_path = random.choice(self.df.iloc[idx].dishes.split(','))
        image = Image.open(f"{self.images_path}/{img_path}/rgb.png").convert('RGB')
        image = self.transforms(image=np.array(image))["image"]
        return {
            "mega_dish_id": mega_dish_id,
            "total_mass": total_mass,
            "total_calories": total_calories,
            "calories_per_gram": calories_per_gram,
            "ingredient_names": ingredient_names,
            "image": image,
        }
    
    def create_attr_tensor(self, batch, attr):
        return torch.Tensor([item[attr] for item in batch])
    
    def get_collate_fn(self, batch):
        mega_dish_id = [item["mega_dish_id"] for item in batch]
        ingredient_names = [item["ingredient_names"] for item in batch]
        images = torch.stack([item["image"] for item in batch])
        tokenized_input = self.tokenizer(ingredient_names,
                                    return_tensors="pt",
                                    padding="max_length",
                                    truncation=True)
        return {
            "mega_dish_id": mega_dish_id,
            "total_mass": self.create_attr_tensor(batch, "total_mass"),
            "total_calories": self.create_attr_tensor(batch, "total_calories"),
            "calories_per_gram": self.create_attr_tensor(batch, "calories_per_gram"),
            "image": images,
            "input_ids": tokenized_input["input_ids"],
            "attention_mask": tokenized_input["attention_mask"],
        }

    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, collate_fn=self.get_collate_fn, *args, **kwargs)
    
    def get_transforms(self):
        cfg = self.image_cfg
        if self.type == "train":
            transforms = A.Compose(
                [
                    A.SmallestMaxSize(
                        max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                    A.RandomCrop(
                        height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                    A.Affine(scale=(0.8, 1.2),
                            rotate=(-15, 15),
                            translate_percent=(-0.1, 0.1),
                            shear=(-10, 10),
                            fill=0,
                            p=0.8),
                    A.CoarseDropout(
                        num_holes_range=(2, 8),
                        hole_height_range=(int(0.07 * cfg.input_size[1]),
                                        int(0.15 * cfg.input_size[1])),
                        hole_width_range=(int(0.1 * cfg.input_size[2]),
                                        int(0.15 * cfg.input_size[2])),
                        fill=0,
                        p=0.5),
                    A.ColorJitter(brightness=0.2,
                                contrast=0.2,
                                saturation=0.2,
                                hue=0.1,
                                p=0.7),
                    A.Normalize(mean=cfg.mean, std=cfg.std),
                    A.ToTensorV2(p=1.0)
                ],
                seed=self.seed,
            )
        else:
            transforms = A.Compose(
                [
                    A.SmallestMaxSize(
                        max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                    A.CenterCrop(
                        height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                    A.Normalize(mean=cfg.mean, std=cfg.std),
                    A.ToTensorV2(p=1.0)
                ],
                seed=self.seed,
            )

        return transforms
