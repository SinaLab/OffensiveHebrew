import torch
import os
import argparse
import logging
import sys
import pickle
from torch.utils.data import DataLoader
from Eval.model import BertClassifier
from Eval.trainer import BertTrainer
from Eval.utils import parse_data_files, set_seed
from Eval.data import DefaultDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path", type=str, required=True, help="Output path",
    )

    parser.add_argument(
        "--eval_path", type=str, required=True, help="Path to eval data",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to checkpoints data",
    )

    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0], help="GPU IDs to train on",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size",
    )

    parser.add_argument(
        "--num_workers", type=int, default=0, help="Dataloader number of workers",
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="Seed for random initialization",
    )

    parser.add_argument(
        "--max_epochs", type=int, default=20, help="Number of model epochs",
    )

    parser.add_argument(
        "--bert_model",
        type=str,
        default="avichr/AlephBERT",
        help="BERT model",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate",
    )

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        force=True
    )

    # Set the seed for randomization
    set_seed(args.seed)

    # Get the datasets and vocab for tags and tokens

    # CHANGES:
    # 12K sentences, all in one file
    # (args.train_path, )
    # datasets[0]
    # vocab is not important, ignore vocab variable
    
    datasets, vocab = parse_data_files((args.eval_path ))
 
    # In eval.py
    with open(os.path.join(args.checkpoint_path, "tag_vocab.pkl"), "rb") as fh:
        vocab = pickle.load(fh)
        
    # From the datasets generate the dataloaders
    # we only care about datasets[0]
    datasets = [
        DefaultDataset(
            segments=dataset, vocab=vocab, bert_model=args.bert_model
        )
        for dataset in datasets
    ]

    shuffle = (True, False, False)
    # CHANGE:
    # eval_dataloader only (return value)
    eval_dataloader = DataLoader(
        dataset=datasets[0],
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=datasets[0].collate_fn,
    )

    # Initialize the model
    # CHANGES:
    # - Remove dropout
    # - Make sure you are passing the right args.bert_model
    # - Load vocabs (we need to save vocab during training)
    # - In train.py
    model = BertClassifier(
        bert_model=args.bert_model, num_labels=len(vocab)
    )

    # ADD:
    # Init model from checkpoint
    # checkpoint_path: path/to/model
    # Add args.checkpoint_path to the argparse
    device = None if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(os.path.join(args.checkpoint_path, "model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(gpu) for gpu in range(len(args.gpus))]
        )
        model = torch.nn.DataParallel(model, device_ids=range(len(args.gpus)))
        model = model.cuda()

    # Initialize the optimizer
    # REMOVE
   # optimizer = torch.optim.AdamW(lr=args.learning_rate, params=model.parameters())

    # Initialize the loss function
    loss = torch.nn.CrossEntropyLoss()

    # Initialize the trainer
    # Update:
    # - Remove optimizer
    # - Remove train_dataloader
    # - Remove val_dataloader
    # - Remove test_dataloader
    # - Remove max_epochs
    classifier = BertTrainer(
        model=model,
        loss=loss,
        output_path=args.output_path,
    )

    # Remove
    #classifier.train()

    # ADD:
    segments, _ = classifier.eval(eval_dataloader)
    classifier.save_predictions(segments, "predictions.csv")

    # ADD:
    classifier.compute_metrics(segments)

    return


if __name__ == "__main__":
    main(parse_args())
