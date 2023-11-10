from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import BertTokenizer
from functools import partial


class Segment:
    def __init__(self, text=None, label=None, pred=None):
        """
        Token object to hold token attributes
        :param text: str
        :param label: str
        :param pred: str
        """
        self.text = text
        self.label = label
        self.pred = pred


class BertSeqTransform:
    def __init__(self, bert_model, vocab, max_seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.encoder = partial(
            self.tokenizer.encode,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=True
        )
        self.max_seq_len = max_seq_len
        self.vocab = vocab

    def __call__(self, segment):
        subwords = self.encoder(segment.text)
        label = self.vocab[segment.label]
        mask = torch.ones(len(subwords))
        return torch.LongTensor(subwords), label, mask


class DefaultDataset(Dataset):
    def __init__(
        self,
        segments=None,
        vocab=None,
        bert_model="avichr/heBERT",
        max_seq_len=512,
    ):
        """
        The dataset that used to transform the segments into training data
        :param examples: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
                         You can get generate examples from -- arabiner.data.dataset.parse_conll_files
        :param vocab: vocab object containing indexed tags and tokens
        :param bert_model: str - BERT model
        :param: int - maximum sequence length
        """
        self.transform = BertSeqTransform(bert_model, vocab, max_seq_len=max_seq_len)
        self.segments = segments
        self.vocab = vocab

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, item):
        subwords, label, mask = self.transform(self.segments[item])
        return subwords, label, mask, self.segments[item]

    def collate_fn(self, batch):
        """
        Collate function that is called when the batch is called by the trainer
        :param batch: Dataloader batch
        :return: Same output as the __getitem__ function
        """
        subwords, labels, masks, segments = zip(*batch)

        # Pad sequences in this batch
        # subwords and tokens are padded with zeros
        # tags are padding with the index of the O tag
        subwords = pad_sequence(subwords, batch_first=True, padding_value=0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return subwords, torch.LongTensor(labels), torch.FloatTensor(masks), segments
