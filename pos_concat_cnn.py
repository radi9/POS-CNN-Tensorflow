import tensorflow as tf
import numpy as np


class POSConcatCNN(object):
    def __init__(self, args):
        # prepare
        self.input_x_text = tf.placeholder(tf.int32, [None, args.seq_length], name="input_x_text")
        self.input_x_tag = tf.placeholder(tf.int32, [None, args.seq_length], name="input_x_tag")
        self.input_y = tf.placeholder(tf.float32, [None, args.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # keep track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # word embedding layer
        with tf.name_scope("word-embedding"):
            # make an embedding matrix
            emb_dim = len(args.vocab[args.vocab.keys()[0]][1])
            emb_mat = np.random.rand(args.vocab_size, emb_dim) * 0.1  # scale down to [0, 0.1]
            for word, (idx, emb_vec) in args.vocab.iteritems():
                emb_mat[idx] = emb_vec
            # make a word embedding variable
            W = tf.Variable(tf.convert_to_tensor(emb_mat, dtype=tf.float32), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x_text)

        # pos embedding layer
        with tf.name_scope("pos-embedding"):
            pos_emb_dim = len(args.pos_vocab[args.pos_vocab.keys()[0]][1])
            pos_emb_mat = np.random.rand(args.pos_vocab_size, pos_emb_dim) * 0.1  # scale down to [0, 0.1]
            for tag, (idx, pos_emb_vec) in args.pos_vocab.iteritems():
                pos_emb_mat[idx] = pos_emb_vec
            W_pos = tf.Variable(tf.convert_to_tensor(pos_emb_mat, dtype=tf.float32), name="W_pos")
            self.embedded_tags = tf.nn.embedding_lookup(W_pos, self.input_x_tag)

        # concatenate word embedding and pos embedding
        self.embedded_concat = tf.concat(2, [self.embedded_chars, self.embedded_tags])
        concat_dim = emb_dim + pos_emb_dim

        # expand to the 4th dimension
        self.embedded_concat = tf.expand_dims(self.embedded_concat, -1)

        # convolution + max-pool layer
        pooled_outputs = []
        for i, filter_size in enumerate(args.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # convolution
                filter_shape = [filter_size, concat_dim, 1, args.num_filters]
                W = tf.Variable(tf.random_uniform(filter_shape, -0.01, 0.01, seed=args.seed), name="W")
                b = tf.Variable(tf.constant(args.bias, shape=[args.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_concat,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # activation
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # max-pooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, args.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # combine pooled features
        num_filters_total = args.num_filters * len(args.filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # dropout layer
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            # TODO: implement custom dropout

        # output layer (fully-connected layer included)
        with tf.name_scope("output"):
            filter_shape_out = [num_filters_total, args.num_classes]
            W = tf.Variable(tf.random_uniform(filter_shape_out, -0.01, 0.01, seed=args.seed), name="W")
            b = tf.Variable(tf.constant(args.bias, shape=[args.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # calculate accuracy
        with tf.name_scope("accuracy"):
            self.targets = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, self.targets)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # calculate loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y)
            self.loss = tf.reduce_mean(losses) + args.l2_reg_lambda * l2_loss
            # TODO: calculate the mean neg log likelihood instead

        # train and update
        with tf.name_scope("update"):
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

            # l2 norm clipping
            self.weight_clipping_op = []
            self.vars_for_clipping = []
            for var in tf.trainable_variables():
                if var.name.startswith('output/W'):
                    self.vars_for_clipping.append(tf.nn.l2_loss(var))
                    updated_var = tf.clip_by_norm(var, args.l2_limit)
                    self.weight_clipping_op.append(tf.assign(var, updated_var))
