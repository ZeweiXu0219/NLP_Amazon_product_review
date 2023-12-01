import torch
import torch.nn as nn
import torch.nn.functional as F


class SLCABG(nn.Module):
    def __init__(self, Embed_size, sentence_length):
      super(SLCABG, self).__init__()
      self.Embed_size = Embed_size
      # self.word_embeddings = nn.Embedding.from_pretrained(word_vectors)#load the pretrained word vectors
      # self.word_embeddings.weight.requires_grad = False
      self.before_lstm_fc = nn.Linear(768, sentence_length)
      self.lstm = nn.GRU(sentence_length, 64, batch_first=True, bidirectional=True, dropout=0.2)
      self.weight_W = nn.Parameter(torch.Tensor(128, 128))
      self.weight_proj = nn.Parameter(torch.Tensor(128, 1))
      self.fc = nn.Linear(128, 2)
      nn.init.uniform_(self.weight_W, -0.1, 0.1)
      nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, x):
      # print(x.shape)
      # embed_x = self.word_embeddings(x)
      # new_x = self.before_lstm_fc(x)
      if self.Embed_size != 768:
        x = self.before_lstm_fc(x)
      new_x = x.permute(0, 2, 1)
      # print(embed_x.shape)
      out, _ = self.lstm(new_x)
      # print(out.shape)
      u = torch.tanh(torch.matmul(out, self.weight_W))
      att = torch.matmul(u, self.weight_proj)
      # print(att)
      att_score = F.softmax(att, dim=1)
      #print(att_score)
      scored_x = out * att_score
      feat = torch.sum(scored_x, dim=1)
      out = self.fc(feat)
      return out