# QDP-FSL

## Introduction

QDP-FSL is a quantum defect prediction (QDP) model designed for quantum software. It leverages a pre-trained code model to understand quantum code semantics and incorporates few-shot learning (FSL) to tackle limited defective samples. Addressing unique challenges in quantum software, QDP-FSL surpasses traditional static analysis methods, advancing quantum software quality assurance.

------

## Directory Structure

```bash
├── Ours/                  # Our results
├── QChecker/              # QChecker tools results
├── code/                  # Main code folder
│   ├── pretrained_models/ # Pretrained model folder
│   │   └── codebert_base/ # Microsoft CodeBERT model
│   ├── pre_data.py        # Data preprocessing script
│   ├── main_fsl.py        # Main program entry point
│   └── ...                # Other auxiliary scripts
├── source/                # Sample data folder
│   ├── 0/             	   # Clean samples
│   ├── 1/                 # Defective samples
├── plot.py                # Visualization script
├── pre_train.sh           # Pretraining script
├── trainFSL.sh            # Automated training script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

------

## Quick Start

### Environment Setup

1. **Install Python Dependencies**

   - This project requires Python >= 3.9.

   - Install the required dependencies using the following command:

     ```
     pip install -r requirements.txt
     ```

2. **Download Microsoft CodeBERT Pretrained Model**

   - Visit [Microsoft CodeBERT GitHub Page](https://github.com/microsoft/CodeBERT).

   - Download the 

     ```
     codebert-base
     ```

     Place the downloaded files in the 

     ```
     code/pretrained_models/codebert_base/
     ```

      directory. The folder structure should look like this:

     ```bash
     code/pretrained_models/codebert_base/
     ├── pytorch_model.bin
     ├── config.json
     ├── special_tokens_map.json
     ├── vocab.json
     ├── tokenizer_config.json
     └── merges.txt
     ```

------

### Data Preparation

1. Place the raw sample data for training and detection into the `source/` directory.

2. Navigate to the `code` folder and run the following command to preprocess the data:

   ```
   python pre_data.py
   ```

------

### Training and Testing

1. Use the following command to start training and testing (modify parameters as needed):

   ```bash
   python main_fsl.py \
       --tokenizer_name=./pretrained_models/codebert_base \
       --model_name_or_path=./pretrained_models/codebert_base \
       --do_train \
       --do_test \
       --num_train_epochs 20 \
       --block_size 256 \
       --early_stopping_patience 4 \
       --min_delta 1e-5 \
       --learning_rate 6e-5 \
       --weight_decay 5e-7 \
       --scheduler_gamma 0.1 \
       --max_grad_norm 1.0 \
       --n_way 2 \
       --n_shot 7 \
       --n_query 1 \
       --n_tasks_per_epoch 200 \
       --n_valid_per_epoch 200 \
       --num_folds 4 \
       --num_times 10 \
       --seed 0 2>&1 | tee train.log
   ```

2. **Parameter Descriptions**:

   - `--tokenizer_name`: Path to the tokenizer
   - `--model_name_or_path`: Path to the pretrained model
   - `--do_train`: Enable training
   - `--do_test`: Enable testing
   - `--early_stopping_patience`: Early stopping patience threshold
   - Other parameters are described in the script.

------

## Logs and Results

- Training logs are saved to `train.log`.
- Results are output to the console and designated output directories.

------

## Notes

1. **Model Path**: Ensure `--tokenizer_name` and `--model_name_or_path` point to the correct pretrained model directory.
2. **Data Format**: The raw data must adhere to the format expected by `pre_data.py`.
3. **Device Compatibility**: It is recommended to run the project in a GPU-supported environment.
