import numpy as np
import tensorflow as tf
import iris
import matplotlib.pyplot as plt
from neural_network_decision_tree import *

np.random.seed(1943)
tf.set_random_seed(1943)

# Input data
x = iris.feature[:, :]
y = iris.label
d = x.shape[1]
num_class = y.shape[1]

# Model params
model_name = "new"
cut = 2
rotate_size = 4
learn_rate = 0.01
train_prob = 1.0
test_prob = 1.0
use_rotate = True
use_cnn = False
k = 3

# Training params
batch_size = 25
run = int(10e3)

args = {"use_rotate": use_rotate,
        "rotate_size": rotate_size,
        "use_cnn": use_cnn,
        "k": k}

# Define the session
sess = tf.InteractiveSession()

# Define the model
x_ph, y_ph, keep_prob, y_pred, train_step, loss = get_model(model_name, d, cut, num_class, learn_rate, args)

# Initialize the variables
sess.run(tf.global_variables_initializer())

# Train the model
for i in range(run):
    batch_idx = np.random.randint(0, x.shape[0], batch_size)
    x_batch = x[batch_idx]
    y_batch = y[batch_idx]
    _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: x_batch, y_ph: y_batch, keep_prob: train_prob})
    if i % 200 == 0:
        loss_all = sess.run([loss], feed_dict={x_ph: x, y_ph: y, keep_prob: test_prob})
        print(loss_all)
print('error rate %.2f' % (1 - np.mean(np.argmax(y_pred.eval(feed_dict={x_ph: x, keep_prob: test_prob}), axis=1) == np.argmax(y, axis=1))))

