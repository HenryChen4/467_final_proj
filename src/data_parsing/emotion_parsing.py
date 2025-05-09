import os
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from src.data_parsing.parsing_base import Parser_Base

class Emotion_Dataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.dataframe = df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = self.dataframe.iloc[index]['statement']
        label = self.dataframe.iloc[index]['status']
        return text, label

class Emotion_Data_Parser(Parser_Base):
    def __init__(self, config):
        super().__init__(config)

        self.labels = [
            'Normal',
            'Depression',
            'Suicidal',
            'Anxiety',
            'Stress',
            'Bipolar',
            'Personality disorder'
        ]

        self.root_dir = config.root_dir
        self.train_split = config.train_split
        self.test_split = config.test_split
        self.val_split = config.val_split
        self.seed = config.seed
        self.max_samples = config.max_samples

        self.csv_data = os.path.join(self.root_dir, config.csv)

        self.vocab = None
        self.label_vocab = None
    
    def balance_data(self, whole_df):
        balanced_dfs = []

        for label in self.labels:
            label_df = whole_df[whole_df['status'] == label]
            sampled_df = label_df.sample(n=self.max_samples, replace=False, random_state=self.seed)

            balanced_dfs.append(sampled_df)

        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
        return balanced_df

    def create_splits(self, whole_df, label):
        label_df = whole_df[whole_df['status'] == label]
        label_df = label_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        n = len(label_df)
        train_end = int(n * 0.7)
        test_end = train_end + int(n * 0.15)

        train_df = label_df.iloc[:train_end]
        test_df = label_df.iloc[train_end:test_end]
        val_df = label_df.iloc[test_end:]

        return train_df, test_df, val_df
    
    def clean_df(self, df):
        return df[df['statement'].notna() & df['statement'].str.strip().astype(bool)]

    def fill_dataloaders(self):
        whole_df = pd.read_csv(self.csv_data, index_col=False) 
        balanced_df = self.balance_data(whole_df=whole_df)
        
        train_df_list = []
        test_df_list = []
        val_df_list = []

        for label in self.labels:
            train_df, test_df, val_df = self.create_splits(whole_df=balanced_df, label=label)
            train_df_list.append(train_df)
            test_df_list.append(test_df)
            val_df_list.append(val_df)

        all_train_df = self.clean_df(pd.concat(train_df_list).reset_index(drop=True))
        all_test_df = self.clean_df(pd.concat(test_df_list).reset_index(drop=True))
        all_val_df = self.clean_df(pd.concat(val_df_list).reset_index(drop=True))

        train_dataset = Emotion_Dataset(all_train_df)
        test_dataset = Emotion_Dataset(all_test_df)
        val_dataset = Emotion_Dataset(all_val_df)

        torch_generator = torch.Generator()
        torch_generator.manual_seed(self.seed)

        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, generator=torch_generator)
        test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, generator=torch_generator)
        val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, generator=torch_generator)

        return train_dataloader, test_dataloader, val_dataloader
    
    def build_vocab(self, train_loader, test_loader, val_loader):
        tokenizer = get_tokenizer("basic_english")
        self.tokenizer = tokenizer

        self.all_samples = []
        for batch in train_loader:
            self.all_samples.extend(zip(batch[0], batch[1]))

        for batch in test_loader:
            self.all_samples.extend(zip(batch[0], batch[1]))

        for batch in val_loader:
            self.all_samples.extend(zip(batch[0], batch[1]))

        texts = [x for (x, _) in self.all_samples]
        labels = [y for (_, y) in self.all_samples]

        def yield_input_tokens():
            for text in texts:
                yield tokenizer(text)

        self.vocab = build_vocab_from_iterator(yield_input_tokens(), specials=["<pad>", "<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.label_vocab = build_vocab_from_iterator([[label] for label in labels])

        return self.vocab, self.label_vocab

    def vectorize_dataloaders(self, data_loader):
        data_loader_samples = list(data_loader)

        self.max_len = self.config.max_len
        input_tensors = []
        label_tensors = []

        for text, label in data_loader_samples:
            token_ids = [self.vocab[token] for token in self.tokenizer(text[0])]

            if len(token_ids) >= self.max_len:
                token_ids = token_ids[:self.max_len]
            else:
                token_ids += [self.vocab["<pad>"]] * (self.max_len - len(token_ids))
            input_tensor = torch.tensor(token_ids, dtype=torch.long)

            label_id = self.label_vocab[label[0]]
            label_tensor = torch.tensor(label_id, dtype=torch.long)

            input_tensors.append(input_tensor)
            label_tensors.append(label_tensor)

        inputs = torch.stack(input_tensors)
        labels = torch.stack(label_tensors)
        dataset = TensorDataset(inputs, labels)

        batch_size = self.config.batch_size
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def tokens_to_words(self, data_loader):
        all_decoded_inputs = []
        all_decoded_labels = []

        for batch_inputs, batch_labels in data_loader:
            for token_ids in batch_inputs:
                indices = token_ids.tolist()
                words = self.vocab.lookup_tokens(indices)
                all_decoded_inputs.append(words)

            for label_token_id in batch_labels:
                index = label_token_id.item()
                label = self.label_vocab.lookup_tokens([index])
                all_decoded_labels.append(label[0])

        return all_decoded_inputs, all_decoded_labels