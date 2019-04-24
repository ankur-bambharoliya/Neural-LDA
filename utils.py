import numpy as np

def load_vocab():
    with open('./data/20news/vocab.new') as vocab_file:
        line = vocab_file.readline()
        vocab = []
        while line:
            parts = line.split(" ")
            vocab.append(parts[0])
            line = vocab_file.readline()
    return vocab

def data_set(dataset_file):
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

def align_data(data, vocab_size):
    """fetch input data by batch."""
    data_align = np.zeros((len(data), vocab_size))

    for i, doc in enumerate(data):
      for word_id, freq in doc.items():
        data_align[i, word_id] = freq

    return  data_align