import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

#train test divide
def train_test_divide(x_data, y_label, ratio = 0.8):#mean_data, std_data, skew_data, median_data, amp_acc, amp_lin

    num_sample, period_len, num_type = x_data.shape

    x_data, y_label = shuffle(x_data, y_label)

    train_size = int(num_sample*ratio)
    train_x_data = x_data[:train_size, :]
    test_x_data = x_data[train_size:, :]

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
#DNN
def DNN(x_data, y_label, ensemble_nn_num, save_path, initializer, num_steps, initial_learning_rate, keep, beta):

    tf.reset_default_graph()

    train_x_data, train_y_data, test_x_data, test_y_data = train_test_divide(x_data, y_label)

    num_class = len(np.unique(train_y_data))

    _, num_step, num_input = train_x_data.shape

    train_x_data = train_x_data.reshape(-1, num_step*num_input, order = 'F') #train_x_data = train_x_data.reshape(-1, num_step*num_input)
    test_x_data = test_x_data.reshape(-1, num_step*num_input, order = 'F') #test_x_data = test_x_data.reshape(-1, num_step*num_input)

    l1_dim = 2400
    l2_dim = 2400
    l3_dim = 1600
    l4_dim = 1000
    l5_dim = 500

    X = tf.placeholder(tf.float32, shape = [None, num_step*num_input])
    Y = tf.placeholder(tf.int32, shape = [None])
    Y_one_hot = tf.one_hot(Y, num_class)

    keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('Layer1') as scope:
        L1, w1, b1 = Layer(X, l1_dim, initializer, keep_prob = keep_prob, W_name = 'W1', b_name = 'b1')

    with tf.variable_scope('Layer2') as scope:
        L2, w2, b2 = Layer(L1, l2_dim, initializer, keep_prob = keep_prob, W_name = 'W2', b_name = 'b2')

    with tf.variable_scope('Layer3') as scope:
        L3, w3, b3 = Layer(L2, l3_dim, initializer, keep_prob = keep_prob, W_name = 'W3', b_name = 'b3')

    with tf.variable_scope('Layer4') as scope:
        L4, w4, b4 = Layer(L3, l4_dim, initializer, keep_prob = keep_prob, W_name = 'W4', b_name = 'b4')

    with tf.variable_scope('Layer5') as scope:
        L5, w5, b5 = Layer(L4, l5_dim, initializer, keep_prob = keep_prob, W_name = 'W5', b_name = 'b5')

    with tf.variable_scope('Output') as scope:
        W6 = tf.get_variable('W6', shape = [l5_dim, num_class], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        b6 = tf.Variable(tf.random_normal([num_class]), name = 'b6')
        L6 = tf.matmul(L5, W6) + b6

    with tf.name_scope('Train'):
        hypothesis = tf.nn.softmax(L6, name = 'hypothesis')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = L6, labels = Y_one_hot), name = 'cost')

        #regularization
        #rg_cost = beta*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)+ tf.nn.l2_loss(W3)) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6)
        loss = cost# + rg_cost

        global_step = tf.Variable(0) #count the # of steps starting from 0
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 100, 0.9)

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name = 'optimizer').minimize(loss)

        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

    #create a saver to save the trained networks
    saver = tf.train.Saver(max_to_keep = ensemble_nn_num)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    accuracies = []

    for nn_idx in range(ensemble_nn_num):

        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) #initialze the variables in the network

        for step in range(num_steps):

            l, a, _ = sess.run([loss, accuracy, optimizer], feed_dict = {X: train_x_data, Y: train_y_data, keep_prob: keep})
            if step % 100 == 0:
                print('iteration: %d, cost: %f, accuracy: %f' %(step, l, a))

        acc = sess.run(accuracy, feed_dict = {X: test_x_data, Y: test_y_data, keep_prob: 1.0})
        accuracies.append(acc)
        saver.save(sess, save_path = os.path.join(save_path,'nn' + str(nn_idx)))

    pred_labels = []

    for nn_idx in range(ensemble_nn_num):
        saver.restore(sess, save_path = os.path.join(save_path, "nn" + str(nn_idx)))
        pred = sess.run(hypothesis, feed_dict = {X: test_x_data, Y: test_y_data, keep_prob: 1.0})
        pred_labels.append(pred)

    # Get average of the predictions of NNs
    ensemble_pred_labels = np.mean(pred_labels, axis=0)

    ensemble_correct = np.equal(np.argmax(ensemble_pred_labels, 1), test_y_data)
    ensemble_accuracy = np.mean(ensemble_correct.astype(np.float32))

    # Compute ROC curve and ROC area for each class
    lw = 2
    test_y_data = label_binarize(test_y_data, classes = range(num_class))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(test_y_data[:, i], ensemble_pred_labels[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y_data.ravel(), ensemble_pred_labels.ravel())
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
    plt.show()

    return accuracies, ensemble_accuracy, EER

data_base_path = 'data'
model_save_base_path = 'checkpoints'
activities = ['walk']

#system parameter
num_test = 1

#network parameters
ensemble_nn_num = 1
num_steps = 5*10**2
initial_learning_rate = 0.001
keep = 0.6
beta = 0
initializer = tf.contrib.layers.xavier_initializer()

for activity in activities:

    data_path = os.path.join(data_base_path, activity)
    model_save_path = os.path.join(model_save_base_path, activity)

    #create training and test data as well as the corresponding labels
    interpolated_period_data = pickle.load(open(os.path.join(data_path, 'filtered_interpolation.pkl'), 'rb'))

    data_set = []
    labels = []

    for user_idx, period_data in enumerate(interpolated_period_data):

        for period in period_data:

            labels.append(user_idx)
            data_set.append(period)

    data_set = np.array(data_set)
    labels = np.array(labels)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    test_result = []

    for _ in range(num_test):

        accuracies, ensemble_accuracy, EER = DNN(data_set, labels, ensemble_nn_num, model_save_path, initializer, num_steps, initial_learning_rate, keep, beta)
        print(accuracies, ensemble_accuracy, EER)
        test_result.append([ensemble_accuracy, EER])

    pickle.dump(test_result, open(os.path.join(data_path, 'classification_result.pkl'), 'wb'))


























































'''
    W1 = tf.get_variable('W1', shape = [num_step*num_input, l1_dim], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([l1_dim]), name = 'b1')
    L1 = tf.matmul(X, W1) + b1
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

    W2 = tf.get_variable('W2', shape = [l1_dim, l2_dim], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([l2_dim]), name = 'b2')
    L2 = tf.matmul(L1, W2) + b2
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

    W3 = tf.get_variable('W3', shape = [l2_dim, l3_dim], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([l3_dim]), name = 'b3')
    L3 = tf.matmul(L2, W3) + b3
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob = keep_prob)

    W4 = tf.get_variable('W4', shape = [l3_dim, l4_dim], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([l4_dim]), name = 'b4')
    L4 = tf.matmul(L3, W4) + b4
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.dropout(L4, keep_prob = keep_prob)

    W5 = tf.get_variable('W5', shape = [l4_dim, l5_dim], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([l5_dim]), name = 'b4')
    L5 = tf.matmul(L4, W5) + b5
    L5 = tf.nn.relu(L5)
    L5 = tf.nn.dropout(L5, keep_prob = keep_prob)
'''

#end
