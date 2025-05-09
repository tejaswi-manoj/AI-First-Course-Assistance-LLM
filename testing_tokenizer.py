import torch.nn as nn
import torch
from model import GPTLanguageModel
import argparse
from utils import get_batch, estimate_loss, levenshtein_distance, get_stats,merge,decode_token_basic,encode_text_token_basic
from rouge import Rouge
import string
import tiktoken
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
    parser.add_argument('--tokenization_strategy', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='./models/test.pth')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of Heads in Attention Block')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of Layers in Attention Block')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--testing_file_prompt', type=str, default='./test_data/test_prompt_1.txt')
    parser.add_argument('--testing_file_response', type=str, default='./test_data/test_response_1.txt')
    parser.add_argument('--testing_file_answer', type=str, default='./test_data/test_answer_1.txt')
    parser.add_argument('--training_file', type=str, default='./train_data/cleaned_aifirst_data.txt')
    parser.add_argument('--generate_token_number', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='cleaned_aifirst_data', choices=['cleaned_aifirst_data'], help='dataset')

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



    if (opt.tokenization_strategy == ''):
        # 100 Tokens
        ascii = string.printable
        chars = list(ascii)
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
    elif(opt.tokenization_strategy == 'GPT2'):
        vocab_size = 50257



    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    model = model.to(opt.device)
    model.load_state_dict(torch.load(opt.ckpt))

    if (opt.tokenization_strategy == ''):
        encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
        context = torch.tensor(encode(text_test_prompt), device='cuda:0').unsqueeze(dim=-1)
        number_gen = len(text_test_answer)
        response = decode(model.generate(context, max_new_tokens=number_gen,block_size=opt.block_size)[0].tolist())
    elif(opt.tokenization_strategy == 'GPT2'):
        enc = tiktoken.get_encoding('gpt2')
        context = torch.tensor(enc.encode(text_test_prompt), device='cuda:0').unsqueeze(dim=-1)
        number_gen = len(text_test_answer)
        response = enc.decode(model.generate(context, max_new_tokens=number_gen, block_size=opt.block_size)[0].tolist())

    with open(opt.testing_file_response, "w") as file:
        file.write(response)
    print(levenshtein_distance(text_test_answer, response))
    rouge = Rouge()
    scores = rouge.get_scores(response, text_test_answer)

    print(scores)



if __name__ == "__main__":

    main()

    
