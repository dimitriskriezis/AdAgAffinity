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


def generate_false_pairs(antibodies, antigens):    
    false_abs = []
    false_ads = []
    for i in range(len(antibodies)-1):
        false_abs.append(antibodies[i])
        false_ads.append(antigens[i+1])
    
    return false_abs, false_ads

def split_train_test(antibodies, antigens):
    train_antibodies = antibodies[:int(0.8*len(antibodies))]
    test_antibodies = antibodies[int(0.8*len(antibodies)):]
    train_antigens = antigens[:int(0.8*len(antigens))]
    test_antigens = antigens[int(0.8*len(antigens)):]

    false_train_antibodies, false_train_antigens = generate_false_pairs(train_antibodies, train_antigens)
    final_train_antibodies = train_antibodies + false_train_antibodies
    final_train_antigens = train_antigens + false_train_antigens
    
    false_test_antibodies, false_test_antigens = generate_false_pairs(test_antibodies, test_antigens)
    final_test_antibodies = test_antibodies + false_test_antibodies
    final_test_antigens = test_antigens + false_test_antigens

    train_labels = [1]*(len(train_antibodies)) + [-1]*len(false_train_antibodies)
    test_labels = [1]*len(test_antibodies) + [-1]*len(false_test_antibodies)

    print(len(train_labels), len(train_antibodies))
    print(len(test_labels), len(test_antibodies))
    return final_train_antibodies, final_train_antigens, train_labels, final_test_antibodies, final_test_antigens, test_labels