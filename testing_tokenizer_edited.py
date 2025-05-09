import torch.nn as nn
import torch
from model import GPTLanguageModel
import argparse
from utils import get_batch, estimate_loss, levenshtein_distance, get_stats, merge, decode_token_basic, encode_text_token_basic
from rouge import Rouge
import string
import tiktoken
import os

def parse_option():
    parser = argparse.ArgumentParser('argument for testing')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=65)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--tokenization_strategy', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='./models/test.pth')
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--training_file', type=str, default='./train_data/CNN_dataset_cleaned.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='CNN_dataset_cleaned', choices=['CNN_dataset_cleaned'])
    parser.add_argument('--num_tests', type=int, default=70, help='Number of test prompts')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_option()

    with open(opt.training_file, 'r', encoding='utf-8') as f:
        text = f.read()

    if opt.tokenization_strategy == '':
        ascii = string.printable
        chars = list(ascii)
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    elif opt.tokenization_strategy == 'GPT2':
        vocab_size = 50257
        enc = tiktoken.get_encoding('gpt2')

    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    model = model.to(opt.device)

    checkpoint = torch.load(opt.ckpt, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    rouge = Rouge()

    for i in range(1, opt.num_tests + 1):
        prompt_file = f'./test_data/test_prompt_{i}.txt'
        answer_file = f'./test_data/test_answer_{i}.txt'
        response_file = f'./test_data/test_response_{i}.txt'

        if not os.path.exists(prompt_file) or not os.path.exists(answer_file):
            print(f"Test {i}: Files not found, skipping.")
            continue

        with open(prompt_file, 'r', encoding='utf-8') as f:
            text_test_prompt = f.read()
        with open(answer_file, 'r', encoding='utf-8') as f:
            text_test_answer = f.read()

        if opt.tokenization_strategy == '':
            context = torch.tensor(encode(text_test_prompt), device=opt.device).unsqueeze(dim=-1)
            number_gen = len(text_test_answer)
            response = decode(model.generate(context, max_new_tokens=number_gen, block_size=opt.block_size)[0].tolist())
        elif opt.tokenization_strategy == 'GPT2':
            context = torch.tensor(enc.encode(text_test_prompt), device=opt.device).unsqueeze(dim=-1)
            number_gen = len(text_test_answer)
            response = enc.decode(model.generate(context, max_new_tokens=number_gen, block_size=opt.block_size)[0].tolist())

        with open(response_file, "w") as file:
            file.write(response)

        print(f"=== Test {i} ===")
        print("Levenshtein distance:", levenshtein_distance(text_test_answer, response))
        scores = rouge.get_scores(response, text_test_answer)
        print("ROUGE scores:", scores)

if __name__ == "__main__":
    main()
