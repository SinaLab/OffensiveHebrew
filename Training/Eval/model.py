from torch import nn
from transformers import BertModel
import logging

logger = logging.getLogger(__name__)


class BertClassifier(nn.Module):
    def __init__(self, bert_model="onlplab/alephbert-base", num_labels=2, dropout=0.1):
        super().__init__()

        self.bert_model = bert_model
        self.num_labels = num_labels
        self.dropout = dropout

        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)

    def forward(self, x, mask):
        output = self.bert(x, attention_mask=mask)
        y = self.dropout(output.pooler_output)
        logits = self.linear(y)
        return logits
