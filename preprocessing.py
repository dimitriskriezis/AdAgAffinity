import pandas as pd
import sys
import numpy as np
from collections import defaultdict
import random

class Vocab:
    def __init__(self) -> None:
        self.vocab = {}
        self.counter = 0
        self.frequency = defaultdict(lambda:0)
        self.abagseq = defaultdict(lambda:[])
        self.abagname = defaultdict(lambda:[])
        self.agnameseq = {}

    def make_embeddings(self, sequence):
        result = []
        for aminoacid in sequence:
            if not aminoacid:
                continue
            if aminoacid not in self.vocab:
                self.vocab[aminoacid] = self.counter
                self.counter += 1
            result.append(self.vocab[aminoacid])
        return result

    def make_vocab(self, train_data):
        antibodies = []
        antigens = []
        antigen_names = []
        for index, pair in train_data.iterrows():
            sequence = []
            ab = pair['antibody_chain']
            ag = pair['antigen_chain']
            antigen_name = pair['antigen_name']
            if antigen_name not in self.abagname[ab]:
                self.abagname[ab].append(antigen_name)
            if ag not in self.abagseq[ab]:
                self.abagseq[ab].append(ag)
            antibody_sequence = self.make_embeddings(pair['antibody_chain'])
            antigen_sequence = self.make_embeddings(pair['antigen_chain'])
            if antigen_name not in self.agnameseq:
                self.agnameseq[antigen_name] = ag


def get_negative_pairs(vocab, antibody):
    # all antigens
    candidate_antigens = set(vocab.agnameseq.keys())
    candidate_antigens -= set(vocab.abagname[antibody])
    antibody_sequences = []
    antigen_sequences = []
    labels = []
    for i in range(len(vocab.abagseq[antibody])):
        choice = random.choice(list(candidate_antigens))
        candidate_antigens.remove(choice)
        ag = vocab.agnameseq[choice]
        antibody_sequences.append(vocab.make_embeddings(antibody))
        antigen_sequences.append(vocab.make_embeddings(ag))
        labels.append(0)
    
    return antibody_sequences, antigen_sequences, labels

def make_dataset(vocab, antibodies):
    antibody_sequences = []
    antigen_sequences = []
    labels = []
    for ab in antibodies:
        # make true pairs
        for ag in vocab.abagseq[ab]:
            antibody_sequences.append(vocab.make_embeddings(ab))
            antigen_sequences.append(vocab.make_embeddings(ag))
            labels.append(1)
        #make negative_pairs
        negative_antibodies, negative_antigens, negative_labels = get_negative_pairs(vocab, ab)
        antibody_sequences.extend(negative_antibodies)
        antigen_sequences.extend(negative_antigens)
        labels.extend(negative_labels)
    return antibody_sequences, antigen_sequences, labels
        
        

def split_train_test(vocab):
    keys = list(vocab.abagseq.keys())
    random.shuffle(keys)
    train_antibodies = keys[:int(0.8*len(keys))]
    test_antibodies = keys[int(0.8*len(keys)):]
    train_antibodies, train_antigens, train_labels = make_dataset(vocab, train_antibodies)
    test_antibodies, test_antigens, test_labels = make_dataset(vocab, test_antibodies)
    return train_antibodies, train_antigens, train_labels, test_antibodies, test_antigens, test_labels
    

def preprocess():
    data = pd.read_csv("new_SaDDAb_antibody_antigen_pairs.tsv", sep='\t')
    data = data[['Hchain_1_letter_seq', 'Lchain_1_letter_seq', 'antigen_chain_1_letter_seq', 'antigen_name']]
    data['antibody_chain'] = data['Hchain_1_letter_seq']
    data['antigen_chain'] = data['antigen_chain_1_letter_seq']
    train_data = data[['antibody_chain', 'antigen_chain', 'antigen_name']]
    train_data = train_data.dropna()
    vocab = Vocab()
    vocab.make_vocab(train_data)
    return vocab
    

if __name__ == '__main__':
    preprocess()

    vocab = preprocess()
    print(vocab.vocab)
    train_antibodies, train_antigens, train_labels, test_antibodies, test_antigens, test_labels = split_train_test(vocab)
    print(train_antibodies[0])
    print(train_antigens[0])
    