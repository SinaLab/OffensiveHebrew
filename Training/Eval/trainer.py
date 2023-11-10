import os
import torch
import logging
import csv
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

logger = logging.getLogger(__name__)


class BertTrainer:
    def __init__(
        self,
        model=None,
        max_epochs=50,
        optimizer=None,
        loss=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        output_path=None,
        clip=5,
        patience=5
    ):
        self.model = model
        self.max_epochs = max_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.output_path = output_path
        self.clip = clip
        self.patience = patience
        self.timestep = 0
        self.epoch = 0

    def save(self):
        """
        Save model checkpoint
        :return:
        """
        filename = os.path.join(self.output_path, "model.pt")

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

        logger.info("Saving checkpoint to %s", filename)
        torch.save(checkpoint, filename)

    def compute_metrics(self, segments):
        """
        Compute macro and micro metrics
        :param y_true: List - ground truth labels
        :param y_pred: List - prediucted labels
        :return:
        """
        y_true = [s.label for s in segments]
        y_pred = [s.pred for s in segments]

        logger.info("\n" + classification_report(y_true, y_pred))

        ma_precision, ma_recall, ma_f1, ma_support = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )

        mi_precision, mi_recall, mi_f1, mi_support = precision_recall_fscore_support(
            y_true, y_pred, average="micro"
        )

        metrics = {
            "f1": {
                "macro": ma_f1,
                "micro": mi_f1
            },
            "precision": {
                "macro": ma_precision,
                "micro": mi_precision
            },
            "recall": {
                "macro": ma_recall,
                "micro": mi_recall
            }
        }

        return metrics

    def save_predictions(self, segments, output_filename):
        with open(output_filename, "w") as fh:
            w = csv.writer(fh, delimiter="\t")
            rows = [["Text", "Label", "Prediction"]]
            rows += [(s.text, s.label, s.pred) for s in segments]
            w.writerows(rows)

    def train(self):
        best_val_loss, test_loss = np.inf, np.inf
        num_train_batch = len(self.train_dataloader)

        for epoch_index in range(self.max_epochs):
            self.epoch = epoch_index
            train_loss = 0

            for batch_index, batch in enumerate(self.train_dataloader, 1):
                _, labels, _, logits = self.classify(batch)
                self.timestep += 1
                batch_loss = self.loss(logits, labels)
                batch_loss.backward()
                self.optimizer.step()
                train_loss += batch_loss.item()

                if self.timestep % 10 == 0:
                    logger.info(
                        "Epoch %d | Batch %d/%d | Timestep %d | Loss %f",
                        epoch_index,
                        batch_index,
                        num_train_batch,
                        self.timestep,
                        batch_loss.item()
                    )

            train_loss /= num_train_batch

            logger.info("** Evaluating on validation dataset **")
            segments, val_loss = self.eval(self.val_dataloader)
            val_metrics = self.compute_metrics(segments)

            logger.info(
                "Epoch %d | Timestep %d | Train Loss %f | Val Loss %f | F1 Micro %f",
                epoch_index,
                self.timestep,
                train_loss,
                val_loss,
                val_metrics["f1"]["micro"]
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info("** Validation improved, evaluating test data **")
                segments, test_loss = self.eval(self.test_dataloader)
                self.save_predictions(segments, os.path.join(self.output_path, "predictions.txt"))
                test_metrics = self.compute_metrics(segments)

                logger.info(
                    f"Epoch %d | Timestep %d | Test Loss %f | F1 Micro %f",
                    epoch_index,
                    self.timestep,
                    test_loss,
                    test_metrics["f1"]["micro"]
                )

                self.save()
            else:
                self.patience -= 1

            # No improvements, terminating early
            if self.patience == 0:
                logger.info("Early termination triggered")
                break

    def classify(self, batch, is_train=True):
        """
        Given a dataloader containing segments, predict the tags
        :param dataloader: torch.utils.data.DataLoader
        :param is_train: boolean - True for training model, False for evaluation
        :return: Iterator
                    subwords (B x T x NUM_LABELS)- torch.Tensor - BERT subword ID
                    gold_tags (B x T x NUM_LABELS) - torch.Tensor - ground truth tags IDs
                    tokens - List[arabiner.data.dataset.Token] - list of tokens
                    valid_len (B x 1) - int - valiud length of each sequence
                    logits (B x T x NUM_LABELS) - logits for each token and each tag
        """
        subwords, labels, masks, segments = batch
        self.model.train(is_train)

        if torch.cuda.is_available():
            subwords = subwords.cuda()
            labels = labels.cuda()

        if is_train:
            self.optimizer.zero_grad()
            logits = self.model(subwords, masks)
        else:
            with torch.no_grad():
                logits = self.model(subwords, masks)

        return subwords, labels, segments, logits

    def eval(self, dataloader):
        ptos = dataloader.dataset.transform.vocab.get_itos()
        golds, preds, segments = list(), list(), list()
        loss = 0

        for batch in dataloader:
            _, labels, batch_segments, logits = self.classify(batch, is_train=False)
            batch_loss = self.loss(logits, labels)

            loss += batch_loss
            preds += torch.argmax(logits, dim=1)
            segments += batch_segments

        loss /= len(dataloader)

        # Update segments, attach predicted tags to each token
        for segment, pred in zip(segments, preds):
            segment.pred = ptos[pred]

        return segments, loss

