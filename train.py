import tensorflow as tf
import argparse
import os
from datetime import datetime
from pos_concat_cnn import POSConcatCNN
from base_cnn import BaseCNN
from logger import Logger
import cPickle
import numpy as np
from random import shuffle
import sys


####################
# HELPER FUNCTIONS #
####################


def split_data(dataset, revs, vocab, pos_vocab, test_fold, args):
    if dataset == 'mr':
        return split_mr_data(revs, vocab, pos_vocab, test_fold, args)
    elif dataset == 'sstb':
        return split_sstb_data(revs, vocab, pos_vocab, args)
    return None


def split_mr_data(revs, vocab, pos_vocab, test_fold, args):
    # split train and test
    x_train_total, y_train_total, x_val_total, y_val_total, x_test_total, y_test_total = [], [], [], [], [], []
    for rev in revs:
        text_tokens, tag_tokens, label, fold_num = \
            rev['text_tokens'], rev['tag_tokens'], rev['label'], rev['fold_num']
        text_tokens = [vocab[token][0] for token in text_tokens]
        tag_tokens = [pos_vocab[tag][0] for tag in tag_tokens]
        if fold_num == test_fold:
            x_test_total.append(zip(text_tokens, tag_tokens))
            y_test_total.append(label)
        else:
            x_train_total.append(zip(text_tokens, tag_tokens))
            y_train_total.append(label)

    # shuffle trainset
    x_y_train = zip(x_train_total, y_train_total)
    shuffle(x_y_train, args.seed)
    x_train_total, y_train_total = list(zip(*x_y_train)[0]), list(zip(*x_y_train)[1])

    # split trainset into trainset and valset
    val_size = int(len(x_train_total) * 0.1)
    x_val_total, y_val_total = x_train_total[:val_size], y_train_total[:val_size]
    x_train_total, y_train_total = x_train_total[val_size:], y_train_total[val_size:]

    return x_train_total, y_train_total, x_val_total, y_val_total, x_test_total, y_test_total


def split_sstb_data(revs, vocab, pos_vocab, args):
    x_train_total, y_train_total, x_val_total, y_val_total, x_test_total, y_test_total = [], [], [], [], [], []
    for rev in revs:
        text_tokens, tag_tokens, label, fold_num = \
            rev['text_tokens'], rev['tag_tokens'], rev['label'], rev['fold_num']
        text_tokens = [vocab[token][0] for token in text_tokens]
        tag_tokens = [pos_vocab[tag][0] for tag in tag_tokens]
        if fold_num == 0:
            x_train_total.append(zip(text_tokens, tag_tokens))
            y_train_total.append(label)
        elif fold_num == 1:
            x_test_total.append(zip(text_tokens, tag_tokens))
            y_test_total.append(label)
        elif fold_num == 2:
            x_val_total.append(zip(text_tokens, tag_tokens))
            y_val_total.append(label)

    # shuffle trainset
    x_y_train = zip(x_train_total, y_train_total)
    shuffle(x_y_train, args.seed)
    x_train_total, y_train_total = list(zip(*x_y_train)[0]), list(zip(*x_y_train)[1])
    return x_train_total, y_train_total, x_val_total, y_val_total, x_test_total, y_test_total


##################
# MAIN FUNCTIONS #
##################


def main():
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--filter_sizes', type=str, default='3,4,5',
                        help='Comma-separated filter sizes')
    parser.add_argument('--num_filters', type=int, default=100,
                        help='Number of filters per filter size')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Dropout keep probability')
    parser.add_argument('--l2_reg_lambda', type=float, default=0.5,
                        help='L2 regularizaion lambda')
    parser.add_argument('--l2_limit', type=float, default=3.0,
                        help='L2 norm limit')
    parser.add_argument('--bias', type=float, default=0.01,
                       help='bias initial value for conv, output layer')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch Size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate for optimizer')

    # misc parameters
    parser.add_argument('--allow_soft_placement', type=int, default=1,
                        help='Allow device soft device placement')
    parser.add_argument('--log_device_placement', type=int, default=0,
                        help='Log placement of ops on devices')
    parser.add_argument('--save_dir', type=str, default='runs',
                       help='directory to store checkpointed models')
    parser.add_argument('--model', type=str, default='pos_concat_cnn',
                       help='which model to run')
    parser.add_argument('--dataset', type=str, default='sstb',
                       help='which dataset to use')
    parser.add_argument('--seed', type=int, default=7777,
                       help='seed for randomness')

    args = parser.parse_args()

    # start training
    initiate(args)


def initiate(args):
    # define output directory
    time_str = datetime.now().strftime('%b-%d-%Y-%H-%M')
    out_dir = os.path.abspath(os.path.join(os.path.curdir, args.save_dir, time_str))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # initiate logger
    log_file_path = os.path.join(out_dir, 'log')
    logger = Logger(log_file_path)

    # report parameters
    logger.write("Parameters:")
    for arg in args.__dict__:
        logger.write("{}={}".format(arg.upper(), args.__dict__[arg]))
    logger.write("")

    # load data and fill args
    logger.write("Loading data...")
    if args.dataset == 'mr':
        revs, word_vocab, pos_vocab, num_folds = cPickle.load(open("mr_data", "rb"))
    elif args.dataset == 'sstb':
        revs, word_vocab, pos_vocab, num_folds = cPickle.load(open("sstb_data", "rb"))
    else:
        logger.write("invalid dataset !!")
        sys.exit()

    args.vocab_size = len(word_vocab)
    args.pos_vocab_size = len(pos_vocab)
    args.seq_length = len(revs[0]['text_tokens'])
    args.num_classes = len(revs[0]['label'])
    args.filter_sizes = map(int, args.filter_sizes.split(","))
    args.vocab = word_vocab
    args.pos_vocab = pos_vocab

    # report
    logger.write("Vocabulary Size: {:d}".format(args.vocab_size))
    logger.write("Number of sentences: {:d}".format(len(revs)))
    logger.write("POS Vocabulary Size: {:d}".format(args.pos_vocab_size))
    logger.write("Sequence Length (with padding): {:d}".format(args.seq_length))
    logger.write("Number of Classes: {:d}\n".format(args.num_classes))

    # construct a model
    if args.model == 'pos_concat_cnn':
        model = POSConcatCNN(args)
    elif args.model == 'base_cnn':
        model = BaseCNN(args)
    else:
        logger.write("invalid model name")
        sys.exit()

    # for train summary
    grad_summaries = []
    for g, v in model.grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.merge_summary(grad_summaries)

    loss_summary = tf.scalar_summary("loss", model.loss)
    acc_summary = tf.scalar_summary("accuracy", model.accuracy)

    train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")

    # prepare saver
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.all_variables())

    # define train
    def train_model(x_text, x_tag, y, dropout_prob, writer, log=False):
        feed_dict = {
            model.input_x_text: np.array(x_text),
            model.input_x_tag: np.array(x_tag),
            model.input_y: np.array(y),
            model.dropout_keep_prob: dropout_prob
        }

        _, step, loss, accuracy, summaries = sess.run(
            [model.train_op, model.global_step, model.loss, model.accuracy, train_summary_op],
            feed_dict)
        sess.run(model.weight_clipping_op)  # rescale weight
        writer.add_summary(summaries, step)
        if log:
            time_str = datetime.now().isoformat()
            logger.write("{}: step {}, loss {:g}, acc {:g}".format(time_str, step-1, loss, accuracy))

    # evaluate
    def evaluate_model(x_text, x_tag, y):
        feed_dict = {
            model.input_x_text: np.array(x_text),
            model.input_x_tag: np.array(x_tag),
            model.input_y: np.array(y),
            model.dropout_keep_prob: 1.0
        }
        step, loss, accuracy, predictions, targets = sess.run(
                [model.global_step, model.loss, model.accuracy, model.predictions, model.targets],
                feed_dict)
        return accuracy, loss

    # start a session
    sess_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement, log_device_placement=args.log_device_placement)
    sess = tf.Session(config=sess_conf)
    with sess.as_default():
        # initialize
        logger.write("Starting session...")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)
        current_step = 0
        max_test_acc_list = []

        for fold in range(num_folds):  # for each fold
            max_val_acc, max_test_acc = 0.0, 0.0
            tf.initialize_all_variables().run()  # initialize the model

            # get split dataset
            x_train_total, y_train_total, x_val_total, y_val_total, x_test_total, y_test_total = \
                split_data(args.dataset, revs, args.vocab, args.pos_vocab, fold, args)
            logger.write("Train/Val/Test Split: {:d}/{:d}/{:d}"
                 .format(len(x_train_total), len(x_val_total), len(x_test_total)))

            num_batches = len(x_train_total) / args.batch_size + 1

            for epoch_num in range(args.num_epochs):  # for each epoch
                # train
                for batch_num in range(num_batches):  # for each batch
                    st, end = batch_num * args.batch_size, (batch_num + 1) * args.batch_size
                    x_batch, y_batch = x_train_total[st:end], y_train_total[st:end]
                    if len(x_batch) == 0:
                        continue
                    x_text_batch = [list(zip(*text)[0]) for text in x_batch]
                    x_tag_batch = [list(zip(*text)[1]) for text in x_batch]
                    train_model(x_text_batch, x_tag_batch, y_batch, args.dropout_keep_prob, train_summary_writer)

                curr_step = tf.train.global_step(sess, model.global_step)
                logger.write("\nEvaluate K={} EPOCH={} STEP={}:".format(fold, epoch_num, curr_step))

                # evaluate with valset
                if len(x_val_total) > 0:
                    val_acc, val_loss = evaluate_model([list(zip(*x)[0]) for x in x_val_total],
                                                       [list(zip(*x)[1]) for x in x_val_total], y_val_total)
                    logger.write("VAL - loss {:g}, acc {:g}".format(val_loss, val_acc))

                # evaluate with testset
                if len(x_test_total) > 0:
                    test_acc, test_loss = evaluate_model([list(zip(*x)[0]) for x in x_test_total],
                                                         [list(zip(*x)[1]) for x in x_test_total], y_test_total)
                    logger.write("TEST - loss {:g}, acc {:g}".format(test_loss, test_acc))

                # save the model
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                # update the result
                max_test_acc = test_acc if val_acc > max_val_acc else max_test_acc
                max_val_acc = max(max_val_acc, val_acc)

            logger.write("----------------------------------------------------------------------------")
            logger.write("K={} BEST TEST ACCURACY {:g}".format(fold, max_test_acc))
            logger.write("----------------------------------------------------------------------------")
            max_test_acc_list.append(max_test_acc)

        logger.write("----------------------------------------------------------------------------")
        logger.write("FINAL MEAN ACCURACY {:g}".format(np.mean(max_test_acc_list)))
        logger.write("----------------------------------------------------------------------------")


if __name__ == '__main__':
    main()