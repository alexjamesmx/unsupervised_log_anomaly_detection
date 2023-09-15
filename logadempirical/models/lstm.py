import torch
import torch.nn as nn
from torch.autograd import Variable
from logadempirical.models.utils import ModelOutput
from typing import Optional


class DeepLog(nn.Module):
    def __init__(self,
                 hidden_size: int = 100,
                 num_layers: int = 2,
                 vocab_size: int = 100,
                 embedding_dim: int = 100,
                 dropout: float = 0.5,
                 criterion: Optional[nn.Module] = None):
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        # NOTE  embeddings dimension are (vocab size + 1,300) an then we initialize them to zero
        self.embedding = nn.Embedding(vocab_size + 1, self.embedding_dim)
        torch.nn.init.zeros_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.criterion = criterion

    def forward(self, batch, device='cpu'):
        x = batch['sequential']
        try:
            y = batch['label']
        except KeyError:
            y = None
        x = self.embedding(x.to(device))
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        probabilities = torch.softmax(logits, dim=-1)
        loss = None
        if y is not None and self.criterion is not None:
            loss = self.criterion(logits, y.view(-1).to(device))
        # print(  f"x shape: {x.shape}, x each element shape: {x[0][0].shape}, logits shape: {logits.shape}, probabilities shape: {probabilities.shape}, loss shape: {loss}")
        return ModelOutput(logits=logits, probabilities=probabilities, loss=loss, embeddings=out[:, -1, :])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def predict(self, src, device="cpu"):
        del src['label']
        return self.forward(src, device=device).probabilities

    def predict_class(self, src, top_k=1, device="cpu"):
        del src['label']
        # NOTE
        # where original tensor or probabilities is a tensor of shape (batch_size, vocab_size)
        # values, indices = torch.topk(self.forward(
        #     src, device=device).probabilities, k=top_k, dim=1)
        # print(
        #     f" original tensor : {self.forward(src, device=device).probabilities.shape}")
        # print(f"values with top k : {top_k}, {values}")
        # print("indices: ", indices)

        return torch.topk(self.forward(src, device=device).probabilities, k=top_k, dim=1).indices


class LogAnomaly(nn.Module):
    def __init__(self,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 vocab_size: int = 100,
                 embedding_dim: int = 300,
                 dropout: float = 0.5,
                 criterion: Optional[nn.Module] = None,
                 use_semantic: bool = True):
        super(LogAnomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_semantic = use_semantic
        self.embedding = None
        if not self.use_semantic:
            self.embedding = nn.Embedding(
                self.vocab_size + 1, self.embedding_dim)
            torch.nn.init.uniform_(self.embedding.weight)
            self.embedding.weight.requires_grad = True

        self.lstm0 = nn.LSTM(input_size=self.embedding_dim,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=False)

        self.lstm1 = nn.LSTM(input_size=1,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=False)
        self.fc = nn.Linear(2 * hidden_size, self.vocab_size)
        self.criterion = criterion

    def forward(self, batch, device='cpu'):
        x_quant = batch['quantitative'].to(device)
        if self.use_semantic:
            x_sem = batch['semantic'].to(device)
        else:
            x_sem = self.embedding(batch['sequential'].to(device))
        try:
            y = batch['label']
        except KeyError:
            y = None
        out0, _ = self.lstm0(x_sem)
        out1, _ = self.lstm1(x_quant)

        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        logits = self.fc(multi_out)
        probabilities = torch.softmax(logits, dim=-1)
        loss = None
        if y is not None and self.criterion is not None:
            loss = self.criterion(logits, y.view(-1).to(device))

        return ModelOutput(logits=logits, probabilities=probabilities, loss=loss, embeddings=multi_out)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def predict(self, batch, device="cpu"):
        del batch['label']
        return self.forward(batch, device=device).probabilities

    def predict_class(self, batch, top_k=1, device="cpu"):
        del batch['label']
        return torch.topk(self.forward(batch, device=device).probabilities, k=top_k, dim=1).indices


if __name__ == '__main__':
    deeplog = DeepLog(128, 2, 100, 128)
    loganomaly = LogAnomaly(1, 128, 2, 100, 300)
    sem_inp = torch.rand(64, 100, 300)
    quan_inp = torch.rand(64, 100)
    seq_inp = torch.randint(100, (64, 100))
    print(quan_inp.shape)
    pred_labels = torch.randint(100, (64,))
    class_labels = torch.randint(2, (64,))
    out = loganomaly(sem_inp, quan_inp, y=pred_labels)
    print("LogAnomaly: ", out.loss)
    out = deeplog(seq_inp, pred_labels)
    print("DeepLog: ", out.loss)
