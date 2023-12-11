import pandas as pd
from itertools import product


def generate_amino_acid_pairs(k):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    pairs = [''.join(comb) for comb in product(amino_acids, repeat=k)]
    return pairs

def cksaap_feature(sequence):
    k=2
    amino_acid_pairs = generate_amino_acid_pairs(k)
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
    raw_data_processing("Covid_antibody_antigen_pairs_with_affinity_scores.csv")
