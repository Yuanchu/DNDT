import tensorflow as tf
from functools import reduce
import numpy as np


def get_model(model_name, d, cut, num_class, learn_rate, args):

    # Model has to be "old" or "new"
    if model_name not in ['old', 'new']:
        raise ValueError("Invalid model: %s!\n" % model_name)
    
    # Old model does not use rotation or CNN
    if model_name == "old":
        args["use_rotate"] = False
        args["use_cnn"] = False
    
    # Control flags
    use_rotate = args.get("use_rotate", True)
    use_cnn = args.get("use_cnn", True)
    
    # Parameters for the new model
    k = args.get("k", 3)
    rotate_size = args.get("rotate_size", 5)
    
    # Cut on k features if use CNN
    # Cut on rotate_size features if only use rotation
    # Cut on d features otherwise 
    if model_name == "new":
        if use_cnn:
            num_cut = [cut for _ in range(k)]
        elif use_rotate:
            num_cut = [cut for _ in range(rotate_size)]
        else:
            num_cut = [cut for _ in range(d)]
    else:
        num_cut = [cut for _ in range(d)]

    num_leaf = np.prod(np.array(num_cut) + 1)

    x_ph = tf.placeholder(tf.float32, [None, d])
    y_ph = tf.placeholder(tf.float32, [None, num_class])
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    if use_rotate:
        W1 = tf.Variable(tf.random_uniform((d, rotate_size), -1, 1))
        x_ph_rotated = tf.matmul(x_ph, W1)
    else:
        x_ph_rotated = x_ph

    cut_points_list = [tf.Variable(tf.random_uniform([i])) for i in num_cut]
    leaf_score = tf.Variable(tf.random_uniform([num_leaf, num_class]))

    # import pdb; pdb.set_trace()
    y_pred = nn_decision_tree(x_ph_rotated, cut_points_list, leaf_score, keep_prob, model_name, args, temperature = 0.1)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits = y_pred, onehot_labels = y_ph))

    opt = tf.train.AdamOptimizer(learn_rate)
    train_step = opt.minimize(loss)

    return x_ph, y_ph, keep_prob, y_pred, train_step, loss


def tf_kron_prod(a, b):
    res = tf.einsum('ij,ik->ijk', a, b)
    res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
    return res


def tf_bin(x, cut_points, hidden = 2, temperature = 0.1):
    """
    x is a N-by-1 matrix (column vector)
    cut_points is a D-dim vector (D is the number of cut-points)
    this function produces a N-by-(D + 1) matrix, each row has only one element being one and the rest are all zeros
    """
    D = cut_points.get_shape().as_list()[0]
    W = tf.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1])
    cut_points = tf.contrib.framework.sort(cut_points)  # make sure cut_points is monotonically increasing
    b = tf.cumsum(tf.concat([tf.constant(0.0, shape = [1]), -cut_points], 0))
    h = tf.matmul(x, W) + b

    res = tf.nn.softmax(h / temperature)
    return res


def nn_decision_tree_old(x, cut_points_list, leaf_score, keep_prob, temperature = 0.1):
    # cut_points_list contains the cut_points for each dimension of feature
    binnings = list(map(lambda z: tf_bin(x[:, z[0]:z[0] + 1], z[1], temperature = temperature), enumerate(cut_points_list)))
    leaf = reduce(tf_kron_prod, binnings)
    leaf_dropped = tf.nn.dropout(leaf, keep_prob)
    return tf.matmul(leaf_dropped, leaf_score)


def nn_decision_tree(x, cut_points_list, leaf_score, keep_prob, model_name, args, temperature=0.1):

    # If use old model, fall back to the old method
    if model_name == "old":
        return nn_decision_tree_old(x, cut_points_list, leaf_score, keep_prob, temperature = temperature)
    else:
        # Add optional CNN layer
        use_cnn = args.get("use_cnn", True)
        if use_cnn:
            k = args.get("k", 3)
            d = x.shape[1]
            reshaped = tf.reshape(x, [-1, 1, d])
            conv1 = tf.layers.conv1d(inputs = reshaped, filters = k, kernel_size = 2, padding = "same", activation = tf.nn.softmax)
            binning_input = tf.reshape(conv1, [-1, k])
        else:
            binning_input = x
        
        # Same kron product and dropout
        binnings = list(map(lambda z: tf_bin(binning_input[:, z[0]:z[0] + 1], z[1], temperature = temperature),
            enumerate(cut_points_list)))        
        leaf = reduce(tf_kron_prod, binnings)
        leaf_dropped = tf.nn.dropout(leaf, keep_prob)
        return tf.matmul(leaf_dropped, leaf_score)
