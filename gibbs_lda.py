import numpy as np
import lda
import utils

vocab = utils.load_vocab()

train_set, train_count = utils.data_set("./data/20news/train.feat")

# y=utils.fetch_data(train_set, train_count, np.arange(11000), 2000)


model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(x)
topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('{}'.format(' '.join(topic_words)))