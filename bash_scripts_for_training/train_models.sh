#!/bin/bash

#Experiment 3: Varying Optimizer (SGD vs. Adam for GPT2 and BPE)
#python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-4 --batch_size 256 --optimizer "AdamW" --max_iters 10000 --save_file './models/model_GPT_AdamW.pth'

python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-4 --batch_size 256 --optimizer "SGD" --max_iters 10000 --save_file './models/model_GPT_SGD.pth'>training_output_GPT_SGD.txt 2>&1

python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-4 --batch_size 256 --optimizer "Adam" --max_iters 10000 --save_file './models/model_GPT_Adam.pth'>training_output_GPT_Adam.txt 2>&1


#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --batch_size 256 --optimizer "AdamW" --target_vocab_size 400 --max_iters 1000 --save_file './models/model_BPE_Adam.pth'

#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --batch_size 256 --optimizer "SGD" --target_vocab_size 400 --max_iters 1000 --save_file './models/model_BPE_SGD.pth'


#Experiment 1: GPT2 Strategy -- Varying Learning Rates with fixed number of iterations: 500
python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-4 --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_lr4.pth'>training_output_GPT_lr4.txt 2>&1

python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-5 --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_lr5.pth'>training_output_GPT_lr5.txt 2>&1

python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-3  --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_lr3.pth'>training_output_GPT_lr3.txt 2>&1

python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-2  --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_lr2.pth'>training_output_GPT_lr2.txt 2>&1

python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-1 --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_lr1.pth'>training_output_GPT_lr1.txt 2>&1



#Experiment 2: BPE Strategy -- Varying vocab size 

#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --target_vocab_size 500 --max_iters 600 --save_file './models/model_BPE_500voc.pth'

#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --target_vocab_size 600 --max_iters 600 --save_file './models/model_BPE_600voc.pth'

#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --target_vocab_size 700 --max_iters 600 --save_file './models/model_BPE_700voc.pth'

#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --target_vocab_size 800 --max_iters 600 --save_file './models/model_BPE_800voc.pth'

#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --target_vocab_size 900 --max_iters 600 --save_file './models/model_BPE_900voc.pth'

#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --target_vocab_size 1000 --max_iters 600 --save_file './models/model_BPE_1000voc.pth'
#helps us find the optimal vocab size
















