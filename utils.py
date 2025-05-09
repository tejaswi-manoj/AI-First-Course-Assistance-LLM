import torch
import torch.nn as nn
from torch.nn import functional as F


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def decode_token_basic(ids,vocab_dict):
    tokens = b"".join(vocab_dict[idx] for idx in ids)
    text = tokens.decode('utf-8',errors='replace')
    return text

def encode_text_token_basic(text,merges):
    tokens = list(text.encode('utf-8'))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats,key=lambda p :merges.get(p,float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
# data loading
def get_batch(split,train_data,val_data,opt):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - opt.block_size, (opt.batch_size,))
    x = torch.stack([data[i:i+opt.block_size] for i in ix])
    y = torch.stack([data[i+1:i+opt.block_size+1] for i in ix])
    x, y = x.to(opt.device), y.to(opt.device)
    return x, y


def levenshtein_distance(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)

    # Initialize a matrix to store distances between prefixes of the strings
    matrix = [[0 for _ in range(len_str2 + 1)] for _ in range(len_str1 + 1)]

    # Fill first row and column
    for i in range(len_str1 + 1):
        matrix[i][0] = i
    for j in range(len_str2 + 1):
        matrix[0][j] = j

    # Calculate distances
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,      # Deletion
                matrix[i][j - 1] + 1,      # Insertion
                matrix[i - 1][j - 1] + cost  # Substitution
            )

    return matrix[len_str1][len_str2]

@torch.no_grad()
def estimate_loss(opt,model,train_data,val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(opt.eval_iters)
        for k in range(opt.eval_iters):
            X, Y = get_batch(split,train_data,val_data,opt)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out