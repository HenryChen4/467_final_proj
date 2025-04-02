import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader 

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

        self.csv_data = os.path.join(self.root_dir, config.csv)
    
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
        
        train_df_list = []
        test_df_list = []
        val_df_list = []

        for label in self.labels:
            train_df, test_df, val_df = self.create_splits(whole_df=whole_df, label=label)
            train_df_list.append(train_df)
            test_df_list.append(test_df)
            val_df_list.append(val_df)

        all_train_df = self.clean_df(pd.concat(train_df_list).reset_index(drop=True))
        all_test_df = self.clean_df(pd.concat(test_df_list).reset_index(drop=True))
        all_val_df = self.clean_df(pd.concat(val_df_list).reset_index(drop=True))

        train_dataset = Emotion_Dataset(all_train_df)
        test_dataset = Emotion_Dataset(all_test_df)
        val_dataset = Emotion_Dataset(all_val_df)

        train_dataloader = DataLoader(dataset=train_dataset)
        test_dataloader = DataLoader(dataset=test_dataset)
        val_dataloader = DataLoader(dataset=val_dataset)

        return train_dataloader, test_dataloader, val_dataloader

