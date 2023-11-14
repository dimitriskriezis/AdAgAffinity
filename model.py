import torch
import torch.nn as nn
from transformer import TransformerModel

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.transfomer = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)
    def forward(self, x):
        pass

class SiameseNetwork(nn.Module):
    def __init__(self, ntokens, emsize, nhead, d_hid, nlayers, dropout):
        super(SiameseNetwork, self).__init__()
        self.encoder = Encoder(ntokens, emsize, nhead, d_hid, nlayers, dropout)
    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        return h1, h2



#
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

# train the network
for epoch in range(num_epochs):
    for i, (sent1, sent2, label) in enumerate(train_data):
        sent1 = torch.tensor(sent1, dtype=torch.long)
        sent2 = torch.tensor(sent2, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)
        h1, h2 = net(sent1, sent2)
        loss = criterion(h1, h2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



