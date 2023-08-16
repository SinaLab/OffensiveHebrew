import torch
import random
import numpy as np
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
import pandas as pd
import logging
from comp9312.classify.data import Segment

logger = logging.getLogger(__name__)


def parse_data_files(data_paths):
    """
    Parse CoNLL formatted files and return list of segments for each file and index
    the vocabs and tags across all data_paths
    :param data_paths: tuple(Path) - tuple of filenames
    :return: tuple( [[(token, tag), ...], [(token, tag), ...]], -> segments for data_paths[i]
                    [[(token, tag), ...], [(token, tag), ...]], -> segments for data_paths[i+1],
                    ...
                  )
             List of segments for each dataset and each segment has list of (tokens, tags)
    """
    datasets, labels = list(), list()

    for data_path in data_paths:
        df = pd.read_csv(data_path)
        dataset = [Segment(**kwargs) for kwargs in df.to_dict(orient="records")]
        datasets.append(dataset)
        labels += [segment.label for segment in dataset]

    # Generate vocabs for tags and tokens
    counter = Counter(labels)
    counter = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    label_vocab = vocab(counter)
    return tuple(datasets), label_vocab


def set_seed(seed):
    """
    Set the seed for random intialization and set
    CUDANN parameters to ensure determmihstic results across
    multiple runs with the same seed

    :param seed: int
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
