from torch.utils.data import Dataset
import torch.tensor
import numpy as np
import utils

class NewsGroupDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_file, vocab_size, transform=None):
        self.transform = transform
        self.data, self.count = utils.data_set(dataset_file)
        self.vocab_size = vocab_size


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = np.zeros(self.vocab_size)

        for word_id, freq in self.data[idx].items():
            sample[word_id] = freq

        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(sample, dtype=torch.float32), self.count[idx]