# **AI Foundations Course Assistance LLM (ECE 2806 Final Project)**

This project presents a customized **GPT-based language model** designed to assist students in the **Special Topics Course, ECE 2806: AI Foundations offered by the School of Electrical and Computer Engineering in Georgia Tech**, with particular emphasis on **Convolutional Neural Networks (CNNs)**. It explores the effects of **dataset selection**, **tokenization strategies**, and **optimizer choices** on model performance, culminating in a fine-tuned, CNN-focused assistant.

A comprehensive project report detailing my methodology, the challenges encountered, and the interpretation of the results is available [here](https://drive.google.com/drive/folders/1DpKBwIpUOYw_f03_aHWqT-q3mkURUiYE).

## 🛠️ **Summary of Codebase Changes**

This section outlines all modifications made to the original codebase for the *ECE 2806: AI Foundations Course Assistance LLM* project.

- **Dataset Preprocessing**  
  Created a custom Python script to parse and clean large AI textbook datasets. It structures the data effectively for training.

- **Evaluation Pipeline Improvements**  
  Replaced the original single-file evaluation with a loop over multiple test prompts (`test_prompt_1.txt` to `test_prompt_70.txt`).

- **Optimizer Generalization**  
  Added command-line support for switching between **Adam**, **AdamW**, and **SGD** optimizers. Originally, only **AdamW** was supported.

- **Checkpointing Enhancements**  
  Introduced checkpoint saving every **500 training iterations**. Checkpoints are saved under `./models/` with names like `checkpoint_500.pth`.

- **Resume Training from Checkpoint**  
  Implemented logic to resume training from the **last available checkpoint**. Automatically loads `model_state_dict` and `optimizer_state_dict` if available.

- **Device Portability**  
  Replaced hardcoded GPU usage with a `--device` argument. Code now works seamlessly on both **GPU** and **CPU** environments.

---

## 🚀 **How to Use This Code**

### 1. **Clone the Repository**

```bash
git clone https://github.gatech.edu/tmanoj3/AI_First_Course_Assistance_LLM_Project.git
cd AIFirst_Final
```

### 2. **Install Requirements**

```bash
pip install -r requirements.txt
```

### 3. **Upload Dataset in /train_data**
Upload dataset in the form of a .txt file.

### 4. **Clean dataset**

Replace the dataset to be cleaned in `clean_text.py`.
Run `clean_text.py` to clean the dataset.

```bash
cd train_data
python3 clean_text.py
```
### 5. **Train the Model**

```bash
cd ..
python3 training_tokenizer_edited.py --tokenization_strategy "GPT2" --learning_rate 3e-4 --batch_size 32 --block_size 128 --optimizer "AdamW" --max_iters 10000 --ckpt './models/model_GPT_AdamW_generated_30000_latest.pth' --save_file './models/model_GPT_AdamW_generated_50000_latest.pth' > training_out_GPT_AdamW_generated_50000.txt 2>&1
```
This command starts training the model using GPT-2 tokenization.
Here is how you can modify or extend it:

--ckpt: This allows training to resume from a previously saved checkpoint (both model weights and optimizer state). Omit this flag if training from scratch.

--save_file: Path to where the new model checkpoint will be saved after training.

--max_iters: Controls how many training steps to run.

--learning_rate, --batch_size, --block_size, --optimizer: These can be changed to tune model performance.

⚠️ Important:
To use a different dataset, modify the training_tokenizer_edited.py file manually — specifically, change the path where the training data is loaded. This script is currently hardcoded to read a particular cleaned file from the train_data/ folder.

### 6. **Evaluate the Model**
```bash
python3 testing_tokenizer_edited.py --tokenization_strategy "GPT2" --block_size 128 --ckpt './models/model_GPT_AdamW_generated_10000_latest.pth’>./testing_outputs/testing_output_10000.txt 2>&1
```
This runs inference using the specified model checkpoint.

--ckpt: This is required. It specifies the path to the trained model file.

--block_size: Must match the block size used during training.

--tokenization_strategy: Should be the same strategy used during training (e.g., "GPT2" or "BPE").

The evaluation script will loop through all prompt files in the test_data/ directory (e.g., test_prompt_1.txt to test_prompt_70.txt) and generate corresponding outputs and metrics (ROUGE, Levenshtein distance) in the testing_outputs/ folder.
