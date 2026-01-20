import torch
import numpy as np
from pathlib import Path
import pandas as pd
from io import StringIO

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

def load_distance_matrix(matrix_path:Path):
    if matrix_path is None:
        identity_matrix = [[0 if ending_letter==starting_letter else 1 for ending_letter in alphabet] for starting_letter in alphabet]
        return identity_matrix
    with open(matrix_path, "r") as f:
        lines = f.readlines()
    lines = r"".join([line for line in lines if line[0]!= "#"]).replace("  ", " ").replace("  ", " ")
    blosum_string = StringIO(lines)
    df = pd.read_csv(blosum_string, sep=" ", index_col=0).to_dict()
    #df["X"]["X"] = max(1,df["X"]["X"])
    return np.array([[df[start][end] for end in ALPHABET] for start in ALPHABET])

def generate_random_matrix(output_path:Path):
    # Create a random matrix with integers between 0 and 13
    matrix = torch.randint(0, 14, (len(ALPHABET), len(ALPHABET)), dtype=torch.float) # Use float to allow division later
    # Make it symmetric
    symmetric_matrix = (matrix + matrix.t()) / 2
    # Set diagonal elements to zero
    symmetric_matrix.fill_diagonal_(0)
    # Convert to integer type if needed, but keep as float for generality
    symmetric_matrix = symmetric_matrix.int()
    torch.save(symmetric_matrix.float(), output_path)
    return symmetric_matrix.float()
    
def make_softlabels(distance, beta=1, proba=False, weight=0.2):
    out = []
    if proba:
        distance = np.exp(beta*distance)
    for it, start in enumerate(distance):
        total_distance = np.sum(1/(np.concatenate([start[:it],start[it+1:]]) ))

        distance_summed = (weight/start)/total_distance
        distance_summed[it] = (1-weight)
        out.append(distance_summed)
    return torch.nn.Embedding.from_pretrained(torch.tensor(np.array(out)).clone().detach(),freeze=True)

def make_distance_embedding(distance, alpha, proba=False):
    if proba:
        distance = torch.exp(alpha*distance)
        return torch.nn.Embedding.from_pretrained(torch.tensor(distance).clone().detach(),freeze=True)
    else:
        return torch.nn.Embedding.from_pretrained(torch.tensor(distance).clone().detach(),freeze=True)

def make_all_distances(distance, alpha, proba=False):
    if proba:
        distance = torch.exp(alpha*distance)
    emb = []
    for corr in distance:
        emb.append([sum([dist**alpha for dist in corr])])
    return torch.nn.Embedding.from_pretrained(torch.tensor(emb).clone().detach(),freeze=True)

if __name__ =="__main__":
    generate_random_matrix("/ibmm_scratch/jgut/blosum_distance/matrices/random_matrix.pt")
    
#scaled
BLOSUM60_DISTANCE_NP = load_distance_matrix("/ibmm_scratch/jgut/blosum_distance/matrices/blosum62_distance.mat")
max_blosum_distance = np.max(BLOSUM60_DISTANCE_NP)
BLOSUM60_DISTANCE = torch.tensor(BLOSUM60_DISTANCE_NP)/max_blosum_distance
BLOSUM60_DISTANCE_NP = BLOSUM60_DISTANCE_NP/max_blosum_distance
CROSS_ENTROPY_DISTANCE_NP=np.ones(len(ALPHABET))-np.eye(len(ALPHABET))
CROSS_ENTROPY_DISTANCE=torch.tensor(CROSS_ENTROPY_DISTANCE_NP)
#scaled
RANDOM_DISTANCE = torch.load("/ibmm_scratch/jgut/blosum_distance/matrices/random_matrix.pt")
max_random_distance = np.max(RANDOM_DISTANCE.numpy())
RANDOM_DISTANCE_NP = RANDOM_DISTANCE.numpy()/max_random_distance
RANDOM_DISTANCE = RANDOM_DISTANCE.float()/max_random_distance
