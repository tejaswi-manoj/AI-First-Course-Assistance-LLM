import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel
import argparse
from utils import get_batch, estimate_loss, levenshtein_distance
import string
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Size of blocks to process vocabulary')
    
    parser.add_argument('--max_iters', type=int, default=256,
                        help='Max Iterations of the Training Process')

    parser.add_argument('--eval_interval', type=int, default=256,
                        help='Max Iterations of the Training Process')
    

    # optimization
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--save_file', type=str, default='./models/test.pth')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='Type of Optimizer for Training my Mode;')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of Heads in Attention Block')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of Layers in Attention Block')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--training_file', type=str, default='./train_data/cleaned_aifirst_data.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='cleaned_aifirst_data',choices=['cleaned_aifirst_data'], help='dataset')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_option()

    with open(opt.training_file, 'r', encoding='utf-8') as f:
        text = f.read()

    ascii = string.printable
    chars = list(ascii)

    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    if(opt.ckpt != ''):
        model.load_state_dict(torch.load(opt.ckpt))
    model = model.to(opt.device)


    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)

    for iter in range(opt.max_iters):

        # every once in a while evaluate the loss on train and val sets
        #if iter % opt.eval_interval == 0:
        losses = estimate_loss(opt,model,train_data,val_data)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train',train_data,val_data,opt)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_file = opt.save_file
    torch.save(model.state_dict(), save_file)
if __name__ == "__main__":
    main()