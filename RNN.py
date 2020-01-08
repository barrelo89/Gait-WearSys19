import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.utils import shuffle
from tensorflow.contrib import rnn
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

#LSTM Cell creation
def lstm_cell(num_units):
    cell = tf.nn.rnn_cell.LSTMCell(num_units = num_units, state_is_tuple=True)
    return cell
#train test divide
def train_test_divide(x_data, y_label, ratio = 0.8):#mean_data, std_data, skew_data, median_data, amp_acc, amp_lin

    num_participant = len(x_data)
    train_size = int(num_participant*ratio)

    x_data, y_label = shuffle(x_data, y_label)

    train_x_data = x_data[:train_size]
    test_x_data = x_data[train_size:]

    train_y_data = y_label[:train_size]
    test_y_data = y_label[train_size:]

    return train_x_data, train_y_data, test_x_data, test_y_data
#a fully-connected layer
def Layer(X, num_output, initializer, keep_prob, W_name, b_name):

    _, num_feature = X.shape

    W = tf.get_variable(W_name, shape = [num_feature, num_output], dtype = tf.float32, initializer = initializer)
    b = tf.Variable(tf.random_normal([num_output]), name = b_name)
    L = tf.matmul(X, W) + b
    L = tf.nn.relu(L)
    L = tf.nn.dropout(L, keep_prob = keep_prob)

    return L, W, b
#RNN layer: Unfortunately, RNN does not work well for this data set
def RNN(x_data, y_labels):

    num_class = len(np.unique(y_labels))

    train_x_data, train_y_data, test_x_data, test_y_data = train_test_divide(x_data, y_labels)

    num_data, num_step, num_input = train_x_data.shape
    num_test, _, _ = test_x_data.shape

    #tunable parameters: fully connected
    l1_dim = 1000
    l2_dim = 1000
    #l3_dim = 512
    #l4_dim = 256
    #l5_dim = 64
    train_keep = 0.6
    test_keep = 1.0
    initial_learning_rate = 5*10**(-4)
    num_steps = 1*10**2
    batch_size = 1000
    initializer = tf.contrib.layers.xavier_initializer()

    #tunable parameter: RNN
    num_hidden = 256
    num_cell = 2


    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape = [None, num_step, num_input])
    Y = tf.placeholder(tf.int32, shape = [None])
    Y_one_hot = tf.one_hot(Y, num_class)
    keep_prob = tf.placeholder(tf.float32)

    X_new = tf.unstack(X, num_step, 1)

    #single-layer RNN
    cell = lstm_cell(num_hidden)
    outputs, states = rnn.static_rnn(cell, X_new, dtype =tf.float32)

    #multi-layer RNN
    #multi_cells = rnn.MultiRNNCell([lstm_cell(num_hidden) for _ in range(num_cell)], state_is_tuple=True)
    #outputs, states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
    ###outputs, states = rnn.static_rnn(multi_cells, X_new, dtype=tf.float32)
    #outputs = tf.transpose(outputs, [1, 0, 2])

    #multi-layer
    L1, w1, b1 = Layer(outputs[-1], l1_dim, initializer, keep_prob = keep_prob, W_name = 'W1', b_name = 'b1')
    L2, w2, b2 = Layer(L1, l2_dim, initializer, keep_prob = keep_prob, W_name = 'W2', b_name = 'b2')
    #L3, w3 = Layer(L2, l3_dim, initializer, keep_prob = keep_prob, W_name = 'W3', b_name = 'b3')
    #L4, w4 = Layer(L3, l4_dim, initializer, keep_prob = keep_prob, W_name = 'W4', b_name = 'b4')
    #L5, w5 = Layer(L4, l5_dim, initializer, keep_prob = keep_prob, W_name = 'W5', b_name = 'b5')

    #W6 = tf.get_variable('W6', shape = [l5_dim, num_class], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    #b6 = tf.Variable(tf.random_normal([num_class]), name = 'b6')
    #L6 = tf.matmul(L5, W6) + b6

    W3 = tf.get_variable('W3', shape = [l2_dim, num_class], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([num_class]), name = 'b3')
    L3 = tf.matmul(L2, W3) + b3

    hypothesis = tf.nn.softmax(L3)

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = L6, labels = Y_one_hot))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = L3, labels = Y_one_hot))

    #regularization
    #rg_cost = beta*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)+ tf.nn.l2_loss(W3)) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6)
    loss = cost# + rg_cost

    global_step = tf.Variable(0) #count the # of steps starting from 0
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 100, 0.96)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Training Started')

    for step in range(num_steps):

        num_batch = int(num_data / batch_size)
        avg_cost = 0
        avg_accu = 0

        for idx in range(num_batch):

            batch_train = train_x_data[idx*batch_size:(idx+1)*batch_size]
            batch_train_label = train_y_data[idx*batch_size:(idx+1)*batch_size]

            feed_dict = {X : batch_train, Y : batch_train_label, keep_prob : train_keep}
            acc, c, _ = sess.run([accuracy, cost, optimizer], feed_dict = feed_dict)
            avg_cost += c / num_batch
            avg_accu += acc/ num_batch
            #print('Epoch: ', '%04d' % (step+1), 'Batch idx: ', '%s' %(str(idx) + '/' +str(num_batch)), 'batch cost = ', '{:.9f}'.format(c), 'avg cost = ', '{:.9f}'.format(avg_cost), 'batch accu = ', '{:.9f}'.format(acc), 'avg accu = ', '{:.9f}'.format(avg_accu))

        #print('\n')
        print('Epoch: ', '%04d' % (step+1), 'Batch Size: ', batch_size, 'Initial Learning Rate: ', initial_learning_rate, 'Accuracy: ', avg_accu, 'Cost: ', avg_cost)
        #print('\n')


    num_batch = np.ceil(num_test / batch_size).astype(np.int16)
    avg_accu = 0

    for idx in range(num_batch):

        if idx == num_batch - 1:
            batch_test = test_x_data[idx*batch_size:]
            batch_test_label = test_y_data[idx*batch_size:]
        else:
            batch_test = test_x_data[idx*batch_size:(idx+1)*batch_size]
            batch_test_label = test_y_data[idx*batch_size:(idx+1)*batch_size]

        feed_dict = {X : batch_test, Y : batch_test_label, keep_prob : test_keep}

        acc = sess.run(accuracy, feed_dict = feed_dict)

        avg_accu += acc/ num_batch

    print('Accuracy: ', avg_accu)

    pred = sess.run(hypothesis, feed_dict = {X: test_x_data, Y: test_y_data, keep_prob: 1.0})

    # Compute ROC curve and ROC area for each class
    lw = 2
    test_y_data = label_binarize(test_y_data, classes = range(num_class))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(test_y_data[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y_data.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fnr = 1 - tpr["macro"]
    EER = fpr["macro"][np.nanargmin(np.absolute((fnr - fpr["macro"])))]
    #print('EER: %f' %(EER))

    # Plot all ROC curves
    plt.figure(1)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    #micro roc: binary classification based
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    #macro roc: multiclass classification based
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

    plt.xlabel('False Positive Rate', fontsize = 15)
    plt.xticks([0.0, 0.05, 0.1, 0.15, 0.2], fontsize = 15)
    plt.ylabel('True Positive Rate', fontsize = 15)
    plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0], fontsize = 15)
    #plt.title('Receiver operating characteristic', fontsize = 15)
    plt.legend(loc="lower right", fontsize = 12)
    plt.tight_layout()

    return avg_accu, EER

#create training and test data as well as the corresponding labels
interpolated_period_data = pickle.load(open('data/filtered_interpolation.pkl', 'rb'))

data_set = []
labels = []

for user_idx, period_data in enumerate(interpolated_period_data):

    for period in period_data:

        labels.append(user_idx)
        data_set.append(period)

data_set = np.array(data_set)
labels = np.array(labels)

print(data_set.shape)

#system parameter
num_test = 10

#network parameters
initial_learning_rate = 0.001
num_epoch = 2*10**2
batch_size = 200
when_to_decay = 1000
decay_ratio = 0.9
train_keep = 0.6
test_keep = 1.0
initializer = tf.contrib.layers.xavier_initializer()

model_save_path = 'checkpoints'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

test_result = []

for _ in range(num_test):

    accuracies, EER = RNN(data_set, labels)
    print(accuracies, EER)
    test_result.append(accuracies)

pickle.dump(test_result, open('data/classification_result.pkl', 'wb'))

















#end
