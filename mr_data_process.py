import re
import numpy as np
import cPickle
from nltk.tag import StanfordPOSTagger


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
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def conv_sent_to_vec(str):
    if str == 'pos':
        return [1, 0]
    else:
        return [0, 1]


def read_mr_data(num_folds, fpath='mr/rt-polarity.{}'):
    revs = []
    vocab = {}
    pos_vocab = {}
    max_len = 0
    pos_tagger = StanfordPOSTagger(
        'pos-tag/english-left3words-distsim.tagger',
        'pos-tag/stanford-postagger.jar',
        'utf8', False, '-mx2000m')
    sentiments = ['pos', 'neg']

    for sentiment in sentiments:
        with open(fpath.format(sentiment), "rb") as f:
            tokens_list = []
            label_vec = conv_sent_to_vec(sentiment)

            # read all the lines
            for line in f.read().splitlines():
                tokens = clean_str(line).split()
                tokens_list.append(tokens)

            # pos tagging
            tokens_list_tagged = pos_tagger.tag_sents(tokens_list)

            for tokens_tagged in tokens_list_tagged:
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
                            'label': label_vec,
                            'fold_num': np.random.randint(0, num_folds)}
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
    print "reading mr dataset ..."
    w2v_bin_path = 'GoogleNews-vectors-negative300.bin'
    pos_bin_path = '1billion-pos.bin'
    num_folds = 10  # 10-fold
    revs, vocab, pos_vocab, max_len = read_mr_data(num_folds)
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
    cPickle.dump([revs, emb_vocab, emb_pos_vocab, num_folds], open("mr_data", "wb"))
    print "mr dataset created!"
