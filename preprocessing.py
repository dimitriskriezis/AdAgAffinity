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
        antigen_names = []
        for index, pair in train_data.iterrows():
            sequence = []
            antibody_sequence = self.make_embeddings(pair['antibody_chain'])
            antigen_sequence = self.make_embeddings(pair['antigen_chain'])
            if antibody_sequence and antigen_sequence:
                antibodies.append(antibody_sequence)
                antigens.append(antigen_sequence)
                antigen_names.append(pair['antigen_name'])
        
        return antibodies, antigens, antigen_names
        


def preprocess():
    data = pd.read_csv("new_SaDDAb_antibody_antigen_pairs.tsv", sep='\t')
    all_antigen_names = set(data['antigen_name'])
    data = data[['Hchain_3_letter_seq', 'Lchain_3_letter_seq', 'antigen_chain_3_letter_seq', 'antigen_name']]
    data['antibody_chain'] = data['Hchain_3_letter_seq'] + ' SEP '+ data['Lchain_3_letter_seq']
    data['antigen_chain'] = data['antigen_chain_3_letter_seq']
    train_data = data[['antibody_chain', 'antigen_chain', 'antigen_name']]
    train_data = train_data.dropna()
    vocab = Vocab()
    antibody_sequences, antigen_sequences, antigen_names = vocab.make_vocab(train_data)
    print(len(antibody_sequences), len(antigen_sequences))
    
    dataset = pd.DataFrame()
    dataset['antibody_sequence'] = pd.Series(antibody_sequences)
    dataset['antigen_sequence'] = pd.Series(antigen_sequences)
    dataset['antigen_names'] = pd.Series(antigen_names)


    return vocab.vocab, antibody_sequences, antigen_sequences, data

# dictionary : (Hchain1letter, Lchain1letter) -> antibody sequence
# dictionary : (Hchain1letter, Lchain1letter) -> antigens names it binds to 
# dictionary : antigename => one antigen sequence
# dictionary : antigen 1letter seq -> antigenname
# for every pair (Hchain1letter, Lchain1letter), (antigen1letterseq)
# Make True Pair: get their sequences make a true pair
# I don't know how to make false pairs

def generate_false_pairs(antibodies, antigens, all_antigen_names):    
    # take a pair
    # randomly select another antigen 
    sys.exit()

def split_train_test(antibodies, antigens, all_antigen_names):
    train_antibodies = antibodies[:int(0.8*len(antibodies))]
    test_antibodies = antibodies[int(0.8*len(antibodies)):]
    train_antigens = antigens[:int(0.8*len(antigens))]
    test_antigens = antigens[int(0.8*len(antigens)):]

    false_train_antibodies, false_train_antigens = generate_false_pairs(train_antibodies, train_antigens, all_antigen_names)
    final_train_antibodies = train_antibodies + false_train_antibodies
    final_train_antigens = train_antigens + false_train_antigens
    
    false_test_antibodies, false_test_antigens = generate_false_pairs(test_antibodies, test_antigens, all_antigen_names )
    final_test_antibodies = test_antibodies + false_test_antibodies
    final_test_antigens = test_antigens + false_test_antigens

    train_labels = [1]*(len(train_antibodies)) + [-1]*len(false_train_antibodies)
    test_labels = [1]*len(test_antibodies) + [-1]*len(false_test_antibodies)

    print(len(train_labels), len(train_antibodies))
    print(len(test_labels), len(test_antibodies))
    return final_train_antibodies, final_train_antigens, train_labels, final_test_antibodies, final_test_antigens, test_labels


if __name__ == '__main__':
    preprocess()