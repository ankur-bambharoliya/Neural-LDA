from torch.utils.data import Dataset
import torch.tensor
import numpy as np

class NewsGroupDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_file, vocab_size, transform=None):
        self.transform = transform
        self.data, self.count = self.parse(dataset_file)
        self.vocab_size = vocab_size

    def parse(self, dataset_file):
        data  = []
        word_count = []
        fin = open(dataset_file)
        while True:
            line = fin.readline()
            if not line:
                break
            id_freqs = line.split()
            doc = {}
            count = 0
            for id_freq in id_freqs[1:]:
                items = id_freq.split(':')
                # python starts from 0
                doc[int(items[0]) - 1] = int(items[1])
                count += int(items[1])
            if count > 0:
                data.append(doc)
                word_count.append(count)
        fin.close()
        return data, word_count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = np.zeros(self.vocab_size)

        for word_id, freq in self.data[idx].items():
            sample[word_id] = freq

        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(sample,dtype=torch.float32)