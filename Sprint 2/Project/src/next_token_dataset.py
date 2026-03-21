import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class next_token_dataset(Dataset):
    def __init__(self, tokenizer, texts=None, min_len=3, max_len=10):
        self.tokenizer = tokenizer
        self.min_len = min_len
        self.max_len = max_len
        self.samples = []
        if texts is not None:
            self.load_texts(texts)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx]["context"]).long(),
            torch.tensor(self.samples[idx]["token_to_predict"]).long(),
        )
    
    def get_data_loader(self, *args, **kwargs):
        def collate_fn(batch):
            texts = [torch.tensor(item[0]) for item in batch]
            labels = torch.tensor([item[1] for item in batch])
            lengths = torch.tensor([len(seq) for seq in texts])
            padded_texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return {
                'input_ids': padded_texts, 
                'lengths': lengths,
                'labels': labels,
            }
        return DataLoader(self, collate_fn=collate_fn, *args, **kwargs)
    
    def load_texts(self, texts):
        texts_tokenized = self.tokenizer(texts, max_length=128, truncation=True)
        for line in tqdm(texts_tokenized["input_ids"]):
            for i in range(1, len(line)):
                context = line[:i]
                if len(context) < self.min_len:
                    continue
                if len(context) > self.max_len:
                    context = context[-self.max_len:]

                self.samples.append({
                    "context": context,
                    "token_to_predict": line[i],
                })

    def read_csv(self, filename, text_column="text", *args, **kwargs):
        chunks = []
        for chunk in tqdm(pd.read_csv(filename, chunksize=10000, *args, **kwargs)):
            chunk = chunk.reset_index(drop=True)
            chunk = chunk[chunk[text_column].notna()]
            texts_series = chunk[text_column].fillna('').astype(str)
            texts_series = texts_series.replace('nan', '').str.strip()
            filtered_texts = [text for text in texts_series if text]
            self.load_texts(filtered_texts)
            chunks.append(chunk)
        return self, pd.concat(chunks, ignore_index=True)

