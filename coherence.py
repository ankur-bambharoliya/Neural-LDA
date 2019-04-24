import math
from collections import defaultdict
import numpy as np
import codecs
import utils

window_total = 11268



def load_word_count():
    word_count = {}
    fin = open('./data/20news/train.feat')
    while True:
        line = fin.readline()
        if not line:
            break
        id_freqs = line.split()
        words = []
        for id_freq in id_freqs[1:]:
            items = id_freq.split(':')
            words.append(str(int(items[0])-1))

        for w1_id in range(0, len(words) - 1):
            if words[w1_id] not in word_count:
                word_count[words[w1_id]] = 0
            word_count[words[w1_id]] += 1
            for w2_id in range(w1_id + 1, len(words)):
                bigram = words[w1_id] + "|" + words[w2_id]
                if bigram not in word_count:
                    bigram = words[w2_id] + "|" + words[w1_id]
                if bigram not in word_count:
                    word_count[bigram] = 0

                word_count[bigram] += 1

    return word_count



def calc_topic_coherence(topic_words, vocab, word_count):
    topic_assoc = []
    for w1_id in range(0, len(topic_words)-1):
        target_word = vocab.index(topic_words[w1_id])

        for w2_id in range(w1_id+1, len(topic_words)):
            topic_word = vocab.index(topic_words[w2_id])
            if target_word != topic_word:
                topic_assoc.append(calc_assoc(str(topic_word), str(target_word), word_count))

    return float(sum(topic_assoc))/len(topic_assoc)

def calc_assoc(word1, word2, word_count):
    combined1 = word1 + "|" + word2
    combined2 = word2 + "|" + word1

    combined_count = 0
    if combined1 in word_count:
        combined_count = word_count[combined1]
    elif combined2 in word_count:
        combined_count = word_count[combined2]
    w1_count = 0
    if word1 in word_count:
        w1_count = word_count[word1]
    w2_count = 0
    if word2 in word_count:
        w2_count = word_count[word2]


    if w1_count == 0 or w2_count == 0 or combined_count == 0:
        result = 0.0
    else:
        result = math.log((float(combined_count)*float(window_total))/ \
            float(w1_count*w2_count), 10)
        result = result / (-1.0*math.log(float(combined_count)/(window_total),10))

    return result


def calc_npmi(topic_file, top_words = 10):

    word_count = load_word_count()
    vocab = utils.load_vocab()

    #read the topic file and compute the observed coherence
    topic_coherence = defaultdict(list) # {topicid: [tc]}
    topic_tw = {} #{topicid: topN_topicwords}
    for topic_id, line in enumerate( codecs.open(topic_file, "r", "utf-8")):
        topic_list = line.split()[:top_words]
        topic_tw[topic_id] = " ".join(topic_list)
        # for n in range(10):
        print(topic_list)
        topic_coherence[topic_id].append(calc_topic_coherence(topic_list, vocab, word_count))

    #sort the topic coherence scores in terms of topic id
    tc_items = sorted(topic_coherence.items())
    mean_coherence_list = []
    for item in tc_items:
        topic_words = topic_tw[item[0]].split()
        mean_coherence = np.mean(item[1])
        mean_coherence_list.append(mean_coherence)
        print ("[%.2f] (" % mean_coherence),
        for i in item[1]:
            print ("%.2f;" % i),
        print (")", topic_tw[item[0]])

    #print the overall topic coherence for all topics
    print ("==========================================================================")
    print ("Average Topic Coherence = %.3f" % np.mean(mean_coherence_list))
    print ("Median Topic Coherence = %.3f" % np.median(mean_coherence_list))

calc_npmi('./results/nvm_lda_topics.txt', 8)