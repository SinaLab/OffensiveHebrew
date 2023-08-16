# Hebrew Corpus

This corpus contains offensive language in Hebrew manually annotated. The data includes 15,881 tweets, labeled with one or more of five classes (abusive, hate, violence, pornographic, or non-offensive). The corpus is annonated manually by Arabic-Hebrew bilingual speakers.

## The Corpus

The corpus contains fine grnaular labels: abusive, hate, violence, pornographic, or non-offensive. However, it is class imbalanced, so we mapped the labels (hate, abusive, violence, and pornographic) into one label we called "offensive", resulting into a binary labeled dataset. Class distribution is presented in the tabel below.

| Class          | Sub-class    | Count  |
| -------------- | ------------ | ------ |
| Offensive      | Abusive      | 124    |
|                | Hate         | 631    |
|                | Pornographic | 4      |
|                | Violence     | 454    |
| Not offeinsive | -            | 14,681 |
| Total          |              | 15,881 |

The binary labaled data is also highly imbalanced with 14,681 tweets labeled as not offensive and 1,200 tweets labeled as offensive. To produce a more balanced dataset, we combined the 1,200 offensive tweets with a random sample of 1,300 non-offensive tweets, resulting in a more balanced dataset of 2,500 tweets. We split the 2,500 tweets into training (70%), validation (10%), and test (20%) sets.

A corpus and model for offenive Hebrew
Version: 1.0 (updated on 7/8/2023)


## Corpus Download

The corpus is available in the `data` directory in this repo.

Training set: data/train.csv
Test set: data/test.csv
Validation set: data/val.csv

The dataset is generally small, so we also made available to the community the remining 14,681 non-offensive tweets to encourage others to contribute and improve the data: data/none-offiensive.csv

## Model Download

Huggingface: https://huggingface.co/SinaLab/OffensiveHebrew

## Requirements

At this point, the code is compatible with `Python 3.10.6` and `torchtext==0.14.0`.

Clone this repo

    git clone https://github.com/SinaLab/OffensiveHebrew

This package has dependencies on multiple Python packages. It is recommended to use Conda to create a new environment that mimics the same environment the model was trained in. Provided in this repo `environment.yml` from which you can create a new conda environment using the command below.

    conda env create -f environment.yml

## Model Training

Argument for model traning are listed below. 

    usage: train.py [-h] --output_path OUTPUT_PATH --train_path TRAIN_PATH
        --val_path VAL_PATH --test_path TEST_PATH
        [--bert_model BERT_MODEL] [--gpus GPUS [GPUS ...]]
        [--max_epochs MAX_EPOCHS] [--batch_size BATCH_SIZE]
        [--num_workers NUM_WORKERS] [--data_config DATA_CONFIG]
        [--learning_rate LEARNING_RATE] [--seed SEED]

    optional arguments:
        --output_path OUTPUT_PATH
            Output path (default: None)
        --train_path TRAIN_PATH
            Path to training data (default: None)
        --val_path VAL_PATH
            Path to training data (default: None)
        --test_path TEST_PATH
            Path to training data (default: None)
        --bert_model BERT_MODEL
            BERT model (default: onlplab/alephbert-base)
		--num_workers NUM_WORKERS
            Dataloader number of workers (default: 0)
		--max_epochs MAX_EPOCHS
			Number of model epochs (default: 20)
		--learning_rate LEARNING_RATE
			Learning rate (default: 0.0001)
        --gpus GPUS [GPUS ...]
            GPU IDs to train on (default: [0])
        --batch_size BATCH_SIZE
            Batch size (default: 32)

        --seed SEED           Seed for random initialization (default: 1)

#### Training Hebrew model
For training the model, pass the following arguments:

    python Training\classify\train.py \
        --output_path /path/to/output/dir \
        --train_path /path/to/train.txt \
        --val_path /path/to/val.txt \
        --test_path /path/to/test.txt \
        --batch_size 8 \
    	--seed  1 \
		--max_epochs  10 \
		--batch_size  8 \
		--bert_model  "onlplab/alephbert-base" \
		--num_workers  1 \
		--gpus  [0] 
## Inference

Inference is the process of using a pre-trained model to perform tagging on a new text. To do that, we will
need the following:

#### Model

Note that the model has the following structure and it is important to keep the same structure for inference to work.

    .
    ├── tag_vocab.pkl
    └── model.pt

#### Inference script

provided in the `eval` directory `eval.py` script that performs inference.

The `eval.py` has the following parameters:

    usage: eval.py [-h] --model_path MODEL_PATH --text
                    TEXT [--batch_size BATCH_SIZE]

    optional arguments:
		--bert_model MODEL_PATH
                            Model path for a pre-trained model, for this we you need to download the checkpoint from this repo  (default: None)
							
		--eval_path TEXT  

							Text or sequence to tag, segments will be identified based on periods (default: None)
							
		--batch_size BATCH_SIZE
                            Batch size (default: 32)
							
		--output_path OUTPUT_PATH
    							path where "model.pt" is saved (default: output/model.pt)
	
		--bert_model BERT_MODEL
            BERT model (default: onlplab/alephbert-base)
			
		--num_workers NUM_WORKERS
            Dataloader number of workers (default: 0)
			
		--max_epochs MAX_EPOCHS
			Number of model epochs (default: 20)
			
		--learning_rate LEARNING_RATE
			Learning rate (default: 0.0001)
			
        --gpus GPUS [GPUS ...]
            GPU IDs to train on (default: [0])
			
        --batch_size BATCH_SIZE
            Batch size (default: 32)

        --seed SEED           Seed for random initialization (default: 1)

Example inference command:

    python -u /path/to/AlephBERT/eval/eval.py
           --model_path /path/to/model
           --text "נסראללה: יורש העצר הסעודי ביקש מטראמפ לרצוח אותי ,Negative"

#### Eval script

Optionally, there is `eval.py` script in `eval` directory to evaluate Hebrew data with ground truth data.

    usage: eval.py [-h] --output_path OUTPUT_PATH --model_path MODEL_PATH
                    --data_paths DATA_PATHS [DATA_PATHS ...] [--batch_size BATCH_SIZE]

     optional arguments:
		--bert_model MODEL_PATH
                            Model path for a pre-trained model, for this we you need to download the checkpoint from this repo  (default: None)
							
		--eval_path TEXT  

							Text or sequence to tag, segments will be identified based on periods (default: None)
							
		--batch_size BATCH_SIZE
                            Batch size (default: 32)
							
		--output_path OUTPUT_PATH
    							path where "model.pt" is saved (default: output/model.pt)
	
		--bert_model BERT_MODEL
            BERT model (default: onlplab/alephbert-base)
			
		--num_workers NUM_WORKERS
            Dataloader number of workers (default: 0)
			
		--max_epochs MAX_EPOCHS
			Number of model epochs (default: 20)
			
		--learning_rate LEARNING_RATE
			Learning rate (default: 0.0001)
			
        --gpus GPUS [GPUS ...]
            GPU IDs to train on (default: [0])
			
        --batch_size BATCH_SIZE
            Batch size (default: 32)

## Citation

Nagham Hamad, Mustafa Jarrar, Mohammed Khalilia, Nadim Nashif : [Offensive Hebrew Corpus and Detection using BERT]
(http://www.jarrar.info/publications/).
