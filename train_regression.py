from itertools import combinations_with_replacement
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from itertools import product
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from regression_model import RegressionTransformerNetwork
import matplotlib.pyplot as plt

class AdAgDataset(Dataset):
    def __init__(self,antigens,antibodies,labels):
        self.antigens = antigens
        self.antibodies = antibodies
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,idx):
        label = torch.tensor(self.labels[idx]).float()
        antigen = torch.tensor(self.antigens[idx]).float()
        antibody = torch.tensor(self.antibodies[idx]).float()
        return antibody,antigen,label

class Train():
    def __init__(self,param_dict):
        self.epochs = param_dict['epochs']
        self.batch_size = param_dict['batch_size']
        self.lr = param_dict['lr']
        self.l2 = param_dict['l2']
        self.data = pd.read_pickle(param_dict['data_path'])
        self.folds = param_dict['folds']

        self.model = None
        self.criterion = None
        self.optimizer = None

        self.cross_validation(self.folds)
        
    def get_trained_model(self,train_loader, test_loader,fold):
        self.model = RegressionTransformerSiamese(400,100,2,3,400)
        self.criterion = nn.MSELoss()
        self.model.apply(self.init_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.l2)

        train_losses = []
        test_losses = []
        test_pred = None
        test_labels = None
        for epoch in range(self.epochs):
            train_loss = self.train_iteration(train_loader)
            test_loss, test_pred, test_labels = self.test_iteration(test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print("epoch",epoch)
            print("train_loss",train_loss)
            print("test_loss",test_loss)

        self.create_loss_plot(train_losses,test_losses,"loss_"+str(fold)+".png")
        self.create_scatter_plot(test_pred,test_labels,"scatter_"+str(fold)+".png")
        pd.to_pickle(test_pred, "test_pred"+str(fold)+".pkl")
        pd.to_pickle(test_labels, "test_labels"+str(fold)+".pkl")
        pd.to_pickle(train_loss,"train_loss"+str(fold)+".pkl")
        pd.to_pickle(test_loss,"test_loss"+str(fold)+".pkl")

    def create_loss_plot(self,train_losses,test_losses,name):
        fig,ax = plt.subplots()
        ax.plot(np.arange(self.epochs),train_losses)
        ax.plot(np.arange(self.epochs),test_losses)
        ax.set_ylabel("MSE")
        ax.set_xlabel("Epochs")
        fig.savefig(name)
        
    def create_scatter_plot(self,pred,labels,name):
        fig,ax = plt.subplots()
        ax.scatter(labels,pred)
        ax.set_ylabel("Predicted Affinity")
        ax.set_xlabel("True Affinity")
        fig.savefig(name)

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def train_iteration(self,train_loader):
        loss = 0
        counter = 0
        for antibodies,antigens, labels in train_loader:
            counter += 1
            out = self.model(antibodies,antigens)
            train_loss = self.criterion(out,labels)

            train_loss.backward()
            self.optimizer.step()
            loss += train_loss.detach().item()
        loss = loss/counter
        return loss

    def test_iteration(self,test_loader):
        loss = 0
        counter = 0
        test_pred = []
        test_labels = []
        for antibodies,antigens, labels in test_loader:
            counter += 1
            out = self.model(antibodies,antigens)
            test_loss = self.criterion(out,labels)
            test_labels = test_labels + labels.tolist()
            test_pred = test_pred + out.tolist()

            loss += test_loss.item()
        loss = loss/counter 
        return loss,test_pred,test_labels

    def cross_validation(self,folds):

        for fold in range(folds):
            train_antibodies, train_antigens, train_affinities, test_antibodies, test_antigens, test_affinities = self.train_test_split_multiple()

            train_dataset = AdAgDataset(train_antibodies, train_antigens, train_affinities)
            test_dataset = AdAgDataset(test_antibodies, test_antigens, test_affinities)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size,shuffle=False)

            self.get_trained_model(train_loader,test_loader,fold)


    def train_test_split_multiple(self, test_size=0.3, random_state=None):
        """
        Split multiple arrays into random train and test subsets.

        Parameters:
        - test_size: float, optional (default=0.j)
            Represents the proportion of the dataset to include in the test split.
        - random_state: int or RandomState instajce, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator.

        Returns:
        - Splitting results as a tuple:
            Tuple containing the train-test split of each input array.
        """
        antibodies = self.data['antibody']
        antigens = self.data['Sequence']
        affinities = self.data['Pred_affinity']

        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size should be between 0.0 and 1.0, but got {test_size}")

        antibodies = np.array(antibodies)
        antigens = np.array(antigens)
        affinities = np.array(affinities)

        num_samples = len(affinities)
        indices = np.arange(num_samples)

        if random_state is not None:
            np.random.seed(random_state)

        np.random.shuffle(indices)
        test_size = int(test_size * num_samples)
        test_indices, train_indices = indices[:test_size], indices[test_size:]

        train_antibodies = antibodies[train_indices]
        train_antigens = antigens[train_indices]
        train_affinities = affinities[train_indices]

        test_antibodies = antibodies[test_indices]
        test_antigens = antigens[test_indices]
        test_affinities = affinities[test_indices]

        return train_antibodies, train_antigens, train_affinities, test_antibodies, test_antigens, test_affinities

def generate_amino_acid_pairs(self,k):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    pairs = [''.join(comb) for comb in product(amino_acids, repeat=k)]
    return pairs

def cksaap_feature(self,sequence):
    k=2
    amino_acid_pairs = self.generate_amino_acid_pairs(k)
    feature_vector = [0] * len(amino_acid_pairs)

    for i in range(len(sequence) - k + 1):
        pair = sequence[i:i+k]
        if pair in amino_acid_pairs:
            index = amino_acid_pairs.index(pair)
            feature_vector[index] += 1

    return feature_vector

def raw_data_processing(data_file):
    data = pd.read_csv(data_file)
    data['antibody'] = data.apply(lambda row: f"{row['HC']}_{['LC']}_{['CDRH1']}_{['CDRH2']}_{['CDRH3']}_{['CDRL1']}_{['CDRL2']}_{['CDRL3']}",axis=1)
    embedded_data = data.loc[:,['Sequence','antibody','Pred_affinity']]
    embedded_data['Sequence'] = embedded_data['Sequence'].apply(cksaap_feature)
    embedded_data['antibody'] = embedded_data['antibody'].apply(cksaap_feature)
    embedded_data.to_pickle('embedded_data.pkl')

if __name__ == "__main__":
    #raw_data_processing("../Covid_antibody_antigen_pairs_with_affinity_scores.csv")

    epochs = 10
    batch_size = 256
    lr = 1e-4
    l2 = 1e-4
    data_path = "embedded_data.pkl"
    folds = 3

    params = {
            "epochs":epochs,
            "batch_size":batch_size,
            "lr":lr,
            "l2":l2,
            "data_path":data_path,
            "folds":folds
            }

    Train(params)
