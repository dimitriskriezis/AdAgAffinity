import torch
import torch.nn as nn
from transformer import TransformerModel
from preprocessing import preprocess
import sys

class Encoder(nn.Module):
    def __init__(self, ntokens, emsize, nhead, d_hid, nlayers, dropout):
        super(Encoder, self).__init__()
        self.transfomer = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)
    def forward(self, x):
        return self.transfomer.forward(x)

class SiameseNetwork(nn.Module):
    def __init__(self, ntokens, emsize, nhead, d_hid, nlayers, dropout):
        super(SiameseNetwork, self).__init__()
        self.encoder = Encoder(ntokens, emsize, nhead, d_hid, nlayers, dropout)
    def forward(self, x1, x2):
        x1 = torch.LongTensor(x1).unsqueeze(axis = 1)
        x2 = torch.LongTensor(x2).unsqueeze(axis = 1)
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        h1 = torch.squeeze(h1, axis = 1)
        h2 = torch.squeeze(h2, axis = 1)
        h1 = torch.mean(h1, axis = 0)
        h2 = torch.mean(h2, axis = 0)
        h1 = torch.unsqueeze(h1, axis = 0)
        h2 = torch.unsqueeze(h2, axis = 0)
        
        return h1, h2



vocab, antibodies, antigens  = preprocess()
print(vocab)
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)
net = SiameseNetwork(ntokens, emsize, nhead, d_hid, nlayers, dropout)

# define the loss function and optimizer
criterion = nn.CosineSimilarity()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
num_epochs = 1
# train the network
counter = 0
for epoch in range(num_epochs):
    for ab, ad in zip(antibodies, antigens):
        counter += 1
        print(counter)
        # sent1 = torch.tensor(sent1, dtype=torch.long)
        # sent2 = torch.tensor(sent2, dtype=torch.long)
        # label = torch.tensor(label, dtype=torch.float)
        h1, h2 = net(ab, ad)
        loss = 1 - criterion(h1, h2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        



