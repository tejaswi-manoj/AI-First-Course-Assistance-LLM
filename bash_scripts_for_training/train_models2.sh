#!/bin/bash

python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-5 --optimizer "AdamW" --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_AdamW_lr5.pth'>training_output_GPT_AdamW_lr5.txt 2>&1
python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-4 --optimizer "AdamW" --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_AdamW_lr4.pth'>training_output_GPT__AdamW_lr4.txt 2>&1
python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-3 --optimizer "AdamW" --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_AdamW_lr3.pth'>training_output_GPT__AdamW_lr3.txt 2>&1
python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-2 --optimizer "AdamW" --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_AdamW_lr2.pth'>training_output_GPT__AdamW_lr2.txt 2>&1
python3 training_tokenizer.py --tokenization_strategy "GPT2" --learning_rate 3e-1 --optimizer "AdamW" --batch_size 256 --max_iters 10000 --save_file './models/model_GPT_AdamW_lr1.pth'>training_output_GPT_AdamW_lr1.txt 2>&1



#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --batch_size 256 --optimizer "Adam" --target_vocab_size 400 --max_iters 10000 --save_file './models/model_BPE_Adam.pth'>training_output_BPE_Adam.txt 2>&1
#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --batch_size 256 --optimizer "AdamW" --target_vocab_size 400 --max_iters 10000 --save_file './models/model_BPE_AdamW.pth'>training_output_BPE_AdamW.txt 2>&1
#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --batch_size 256 --optimizer "SGD" --target_vocab_size 400 --max_iters 10000 --save_file './models/model_BPE_SGD.pth'>training_output_BPE_SGD.txt 2>&1


#useless probably:

#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --optimizer "Adam" --target_vocab_size 500 --max_iters 600 --save_file './models/model_BPE_500voc.pth'>training_output_BPE_500voc.txt 2>&1
#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --optimizer "Adam" --target_vocab_size 600 --max_iters 600 --save_file './models/model_BPE_600voc.pth'>training_output_BPE_600voc.txt 2>&1
#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --optimizer "Adam" --target_vocab_size 700 --max_iters 600 --save_file './models/model_BPE_700voc.pth'>training_output_BPE_700voc.txt 2>&1
#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --optimizer "Adam" --target_vocab_size 800 --max_iters 600 --save_file './models/model_BPE_800voc.pth'>training_output_BPE_800voc.txt 2>&1
#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --optimizer "Adam" --target_vocab_size 900 --max_iters 600 --save_file './models/model_BPE_900voc.pth'>training_output_BPE_900voc.txt 2>&1
#python3 training_tokenizer.py --tokenization_strategy "BPE" --learning_rate 3e-4 --optimizer "Adam" --target_vocab_size 1000 --max_iters 600 --save_file './models/model_BPE_1000voc.pth'>training_output_BPE_1000voc.txt 2>&1

















