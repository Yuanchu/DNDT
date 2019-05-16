import argparse
import iris
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neural_network_decision_tree import *

np.random.seed(1943)
tf.set_random_seed(1943)

def get_input():
    return iris.feature[:, :], iris.label

if __name__ == "__main__":

    # Parse args 
    # Model params
    parser = argparse.ArgumentParser(prog = 'Train the mode.')
    parser.add_argument('--model', type = str, default = "old", choices = ["old", "new"], help = 'Model name: old or new.')
    parser.add_argument('--cut', type = int, default = 2, help = 'Number of cut per feature.')
    parser.add_argument('--use_rotate', type = bool, default = True, help = 'Whether or not to include the rotation layer.')
    parser.add_argument('--rotate_size', type = int, default = 4, help = 'Number of neurons in the rotation layer.')
    parser.add_argument('--use_cnn', type = bool, default = True, help = 'Whether or not to include the convolution layer.')
    parser.add_argument('--k', type = int, default = 3, help = 'Output dimension of the convolution.')
    parser.add_argument('--train_prob', type = float, default = 0.9, help = 'Keep probability during the training phase.')

    # Training params
    parser.add_argument('--learn_rate', type = float, default = 1e-2, help = 'Learning rate for the optimization.')
    parser.add_argument('--batch_size', type = int, default = 25, help = 'Number of instances per batch.')
    parser.add_argument('--run', type = int, default = int(1e4), help = 'Number of iterations for training.')

    args = parser.parse_args()

    # Input data from the Iris dataset
    x, y = get_input()
    d = x.shape[1]
    num_class = y.shape[1]

    # Model params
    model_name = args.model
    cut = args.cut
    rotate_size = args.rotate_size
    train_prob = args.train_prob
    test_prob = 1.0
    use_rotate = args.use_rotate
    use_cnn = args.use_cnn
    k = args.k

    # Training params
    learn_rate = args.learn_rate
    batch_size = args.batch_size
    run = args.run

    model_args = {"use_rotate": use_rotate,
                  "rotate_size": rotate_size,
                  "use_cnn": use_cnn,
                  "k": k}

    # Create the session
    sess = tf.InteractiveSession()

    # Define the model
    x_ph, y_ph, keep_prob, y_pred, train_step, loss = get_model(model_name, d, cut, num_class, learn_rate, model_args)

    # Initialize the variables
    sess.run(tf.global_variables_initializer())

    # Train the model
    for i in range(run):
        batch_idx = np.random.randint(0, x.shape[0], batch_size)
        x_batch = x[batch_idx]
        y_batch = y[batch_idx]
        _, loss_e = sess.run([train_step, loss], feed_dict = {x_ph: x_batch, y_ph: y_batch, keep_prob: train_prob})
        if i % 200 == 0:
            loss_all = sess.run([loss], feed_dict = {x_ph: x, y_ph: y, keep_prob: test_prob})
            print(loss_all)
    print('error rate %.2f' % (1 - np.mean(np.argmax(y_pred.eval(feed_dict = {x_ph: x, keep_prob: test_prob}), axis = 1) == np.argmax(y, axis = 1))))

