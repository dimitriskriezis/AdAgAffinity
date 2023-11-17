import torch
import torch.nn as nn
from transformer import TransformerModel
from preprocessing import preprocess, split_train_test
import sys
from eval import eval, roc_plot
from tqdm import tqdm


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
        # h1 = torch.unsqueeze(h1, axis = 0)
        # h2 = torch.unsqueeze(h2, axis = 0)
        
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
criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
num_epochs = 10 
# train the network
train_antibodies, train_antigens, train_labels, test_antibodies, test_antigens, test_labels = split_train_test(antibodies, antigens)
counter = 0
for epoch in tqdm(range(num_epochs)):
    for ab, ad, label in zip(train_antibodies, train_antigens, train_labels):
        counter += 1
        h1, h2 = net(ab, ad)
        loss = criterion(h1, h2, torch.tensor(label))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

similarities = []
for ab, ad in zip(train_antibodies, train_antigens):
    h1, h2 = net(ab, ad)
    h1 = torch.unsqueeze(h1, axis = 0)
    h2 = torch.unsqueeze(h2, axis = 0)
    similarity = nn.CosineSimilarity()(h1, h2).detach().numpy()[0]
    similarities.append(similarity)

f, t, auc = eval(train_labels, similarities)
print("Train auc: ", auc)
# true pairs
similarities = []
for ab, ad in zip(test_antibodies, test_antigens):
    h1, h2 = net(ab, ad)
    h1 = torch.unsqueeze(h1, axis = 0)
    h2 = torch.unsqueeze(h2, axis = 0)
    similarity = nn.CosineSimilarity()(h1, h2).detach().numpy()[0]
    similarities.append(similarity)

f, t, auc = eval(test_labels, similarities)
print("Test auc: ", auc)