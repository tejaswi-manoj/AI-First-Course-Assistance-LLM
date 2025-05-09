import torch.nn as nn
import torch
from model import GPTLanguageModel
import argparse
from utils import levenshtein_distance
from rouge import Rouge
import string

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Size of blocks to process vocabulary')
    parser.add_argument('--vocab_size', type=int, default=65,
                        help='Size of blocks to process vocabulary')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout')


    # model dataset
    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--ckpt', type=str, default='./models/test.pth')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of Heads in Attention Block')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of Layers in Attention Block')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--testing_file_prompt', type=str, default='./test_data/test_prompt_1.txt')
    parser.add_argument('--testing_file_response', type=str, default='./test_data/test_response_1.txt')
    parser.add_argument('--testing_file_answer', type=str, default='./test_data/test_answer_1.txt')
    parser.add_argument('--training_file', type=str, default='./train_data/shakespeare.txt')
    parser.add_argument('--generate_token_number', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='shakespeare', choices=['shakespeare'], help='dataset')

    opt = parser.parse_args()

    return opt
def main():
    opt = parse_option()
    with open(opt.testing_file_prompt, 'r', encoding='utf-8') as f:
        text_test_prompt = f.read()

    with open(opt.testing_file_answer, 'r', encoding='utf-8') as f:
        text_test_answer= f.read()

    with open(opt.training_file, 'r', encoding='utf-8') as f:
        text = f.read()

    ascii = string.printable
    chars = list(ascii)
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}


    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    model = model.to(opt.device)
    model.load_state_dict(torch.load(opt.ckpt))

    context = torch.tensor(encode(text_test_prompt), device='cuda:0').unsqueeze(dim=-1)

    number_gen = len(context)
    response = decode(model.generate(context, max_new_tokens=number_gen,block_size=opt.block_size)[0].tolist())

    with open(opt.testing_file_response, "w") as file:
        file.write(response)
    print(levenshtein_distance(text_test_answer, response))
    rouge = Rouge()
    scores = rouge.get_scores(response, text_test_answer)

    print(scores)



if __name__ == "__main__":

    main()

    
