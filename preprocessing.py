import pandas as pd
import sys
import numpy as np
from collections import defaultdict

class Vocab:
    def __init__(self) -> None:
        self.vocab = {}
        self.counter = 0
        self.frequency = defaultdict(lambda:0)
    
    def make_embeddings(self, sequence):
        result = []
        for aminoacid in sequence.split(" "):
            if not aminoacid:
                continue
            if len(aminoacid) != 3 and len(aminoacid) > 0:
                return []
            if aminoacid not in self.vocab:
                self.vocab[aminoacid] = self.counter
                self.counter += 1
            result.append(self.vocab[aminoacid])
            self.frequency[aminoacid]+=1
        return result

    def make_vocab(self, train_data):
        antibodies = []
        antigens = []
        for index, pair in train_data.iterrows():
            sequence = []
            antibody_sequence = self.make_embeddings(pair['antibody_chain'])
            antigen_sequence = self.make_embeddings(pair['antigen_chain'])
            if antibody_sequence and antigen_sequence:
                antibodies.append(antibody_sequence)
                antigens.append(antigen_sequence)
        
        return antibodies, antigens
        


def preprocess():
    data = pd.read_csv("SaDDAb_antibody_antigen_pairs.tsv", sep='\t')
    data = data[['Hchain_3_letter_seq', 'Lchain_3_letter_seq', 'antigen_chain_3_letter_seq']]
    data['antibody_chain'] = data['Hchain_3_letter_seq'] + ' SEP '+ data['Lchain_3_letter_seq']
    data['antigen_chain'] = data['antigen_chain_3_letter_seq']
    train_data = data[['antibody_chain', 'antigen_chain']]
    train_data = train_data.dropna()
    vocab = Vocab()
    antibody_sequences, antigen_sequences = vocab.make_vocab(train_data)
    print(len(antibody_sequences), len(antigen_sequences))

    return vocab.vocab, antibody_sequences, antigen_sequences