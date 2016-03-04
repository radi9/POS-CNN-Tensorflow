import re
import numpy as np
import cPickle
from nltk.tag import StanfordPOSTagger
import csv


PAD_WORD = '\PAD'


def get_emb_vocab(fpath, vocab):
    with open(fpath, "rb") as f:
        emb_vocab = {}
        hdr = f.readline()
        emb_vocab_size, emb_dim = map(int, hdr.split())
        binary_len = np.dtype('float32').itemsize * emb_dim
        num_pretrained = 0

        # add word predefined word vectors
        for line in xrange(emb_vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            emb_vec = list(np.fromstring(f.read(binary_len), dtype='float32'))
            if word in vocab:
                num_pretrained += 1
                emb_vocab[word] = (vocab[word], emb_vec)

        # add unknown words
        for (word, idx) in vocab.iteritems():
            if word not in emb_vocab:
                emb_vocab[word] = (idx, [np.random.ranf() * 0.5 - 0.25] * emb_dim)

        print '{} entries found in a pretrained set !!'.format(num_pretrained)
        return emb_vocab


def clean_str(string):
    string = re.sub(r"-LRB-", "(", string)
    string = re.sub(r"-RRB-", ")", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_fold_num(split):
    if split == 'train':
        return 0
    elif split == 'test':
        return 1
    elif split == 'dev':
        return 2
    else:
        return -1


def conv_label_to_label_vec(label):
    if label == '0':
        return [1, 0, 0, 0, 0]
    elif label == '1':
        return [0, 1, 0, 0, 0]
    elif label == '2':
        return [0, 0, 1, 0, 0]
    elif label == '3':
        return [0, 0, 0, 1, 0]
    elif label == '4':
        return [0, 0, 0, 0, 1]
    return None


def read_sstb_data(fpath='sstb/sstb_condensed_{}.csv'):
    revs = []
    vocab = {}
    pos_vocab = {}
    max_len = 0
    pos_tagger = StanfordPOSTagger(
        'pos-tag/english-left3words-distsim.tagger',
        'pos-tag/stanford-postagger.jar',
        'utf8', False, '-mx2000m')

    dataset_split = ['train', 'test', 'dev']

    for split in dataset_split:
        with open(fpath.format(split), "rb") as f:
            rdr = csv.reader(f)
            tokens_list = []
            labels = []

            # read all the lines
            for row in rdr:
                tokens = clean_str(row[0]).split()
                tokens_list.append(tokens)
                labels.append(row[1])

            # pos tagging
            tokens_list_tagged = pos_tagger.tag_sents(tokens_list)

            for i in range(len(tokens_list_tagged)):
                tokens_tagged = tokens_list_tagged[i]
                label = labels[i]
                text_tokens = list(zip(*tokens_tagged)[0])
                tag_tokens = list(zip(*tokens_tagged)[1])

                # add each token to vocab
                for token in text_tokens:
                    if token not in vocab:
                        vocab[token] = len(vocab)
                for tag in tag_tokens:
                    if tag not in pos_vocab:
                        pos_vocab[tag] = len(pos_vocab)

                # get max len
                max_len = max(max_len, len(text_tokens))

                # create an entry for the current rev and add to the list
                curr_rev = {'text_tokens': text_tokens,
                            'tag_tokens': tag_tokens,
                            'label': conv_label_to_label_vec(label),
                            'fold_num': get_fold_num(split)}
                revs.append(curr_rev)

    # add padding word
    vocab[PAD_WORD] = len(vocab)
    pos_vocab[PAD_WORD] = len(pos_vocab)

    return revs, vocab, pos_vocab, max_len


def pad_revs(revs, max_len, extra_pad=4):
    keys = ['text_tokens', 'tag_tokens']
    for rev in revs:
        for key in keys:
            tokens = rev[key]
            new_tokens = [PAD_WORD] * extra_pad
            new_tokens.extend(tokens)
            new_tokens.extend([PAD_WORD] * (max_len - len(tokens)))
            new_tokens.extend([PAD_WORD] * extra_pad)
            rev[key] = new_tokens


if __name__ == '__main__':
    print "reading sstb dataset ..."
    w2v_bin_path = 'GoogleNews-vectors-negative300.bin'
    pos_bin_path = '1billion-pos-48.bin'
    num_folds = 1  # sstb doesn't need k-fold cv
    revs, vocab, pos_vocab, max_len = read_sstb_data()
    pad_revs(revs, max_len)
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "pos vocab size: " + str(len(pos_vocab))
    print "max sentence length: " + str(max_len)
    print "loading pre-trained word embedding vectors..."
    emb_vocab = get_emb_vocab(w2v_bin_path, vocab)
    print "loading pre-trained pos embedding vectors..."
    emb_pos_vocab = get_emb_vocab(pos_bin_path, pos_vocab)
    print "embeddings loaded!"
    cPickle.dump([revs, emb_vocab, emb_pos_vocab, num_folds], open("sstb_data", "wb"))
    print "sstb dataset created!"
