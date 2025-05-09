import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm, trange

from src.classifier.classifier_base import Classifier_Base

class LSTM(nn.Module):
    def __init__(self, config, text_vocab):
        super(LSTM, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(len(text_vocab), config.embedding_dim, padding_idx=text_vocab["<pad>"])
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.latent_dim,
                            num_layers=config.lstm_layers,
                            batch_first=True,
                            bidirectional=config.bidirectional)
        
        self.latent_dim = config.latent_dim * 2 if config.bidirectional else config.latent_dim

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        if self.config.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1] 

        return final_hidden

class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super(MLP, self).__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units

        self.architecture = [nn.Linear(input_dim, self.num_hidden_units),
                             nn.ReLU(),
                             nn.Dropout(self.config.dropout_prob)]

        for i in range(self.num_hidden_layers):
            self.architecture.append(nn.Linear(self.num_hidden_units, self.num_hidden_units))
            self.architecture.append(nn.ReLU())
            self.architecture.append(nn.Dropout(self.config.dropout_prob))

        self.architecture.append(nn.Linear(self.num_hidden_units, output_dim))

        self.model = nn.Sequential(*self.architecture)

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)

class Softmax_Classifier(nn.Module):
    def __init__(self, config):
        super(Softmax_Classifier, self).__init__()
        self.config = config
    
    def finish_initialization(self, text_vocab, label_vocab):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.encoder = LSTM(self.config, text_vocab).to(self.device)
        self.classifier = MLP(self.config, 
                              input_dim=self.config.latent_dim * 2 if self.config.bidirectional else self.config.latent_dim,
                              output_dim=len(label_vocab)).to(self.device)
    
    def forward(self, x):
        encoded = self.encoder(x)
        classifications = self.classifier(encoded)
        return classifications

    def train(self, train_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()

        epoch_loss = []

        for epoch in trange(self.config.epochs):
            avg_loss = 0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = self.forward(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()

            print(f"BATCH LOSS: {avg_loss/len(train_loader)}")
            epoch_loss.append(avg_loss/len(train_loader))

        return epoch_loss

    def predict(self, test_loader):
        all_preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)
                logits = self.forward(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())

        return torch.cat(all_preds)
