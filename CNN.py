import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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
def F_Layer(X, num_output, initializer, W_name, b_name, keep_prob):

    _, num_feature = X.shape

    W = tf.get_variable(W_name, shape = [num_feature, num_output], dtype = tf.float32, initializer = initializer)
    b = tf.Variable(tf.random_normal([num_output]), name = b_name)
    L = tf.matmul(X, W) + b
    L = tf.nn.relu(L)
    L = tf.nn.dropout(L, keep_prob = keep_prob)

    return L, W, b
#a convolution layer
def C_Layer(X, L_window_size, L_depth, L_num_kernel, initializer, K_name, b_name, keep_prob):

    K = tf.get_variable(K_name, dtype = tf.float32, shape = [L_window_size, L_window_size, L_depth, L_num_kernel], initializer = initializer)
    b = tf.Variable(tf.random_normal([L_num_kernel]), name = b_name)
    L = tf.nn.conv2d(X, K, strides = [1, 1, 1, 1], padding = 'SAME') + b
    L = tf.nn.relu(L)
    L = tf.nn.max_pool(L, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    L = tf.nn.dropout(L, keep_prob=keep_prob)

    return L
#CNN
def CNN(x_data, y_labels, num_epoch, batch_size, initial_learning_rate, when_to_decay, decay_ratio, train_keep, test_keep):

    num_class = len(np.unique(y_labels))

    train_x_data, train_y_data, test_x_data, test_y_data = train_test_divide(x_data, y_labels)

    num_data, num_step, num_input = train_x_data.shape
    num_test, _, _ = test_x_data.shape

    tf.reset_default_graph()

    #setting placeholders
    X = tf.placeholder(tf.float32, [None, num_step, num_input])
    #to apply kernel (filter), reshape 2D input img into 3D img considering its # of channel
    X_img = tf.reshape(X, [-1, num_step, num_input, 1])
    Y = tf.placeholder(tf.int32, [None])
    Y_one_hot = tf.one_hot(Y, num_class)

    keep_prob = tf.placeholder(tf.float32)

    #parameter matrix initializer
    initializer = tf.contrib.layers.xavier_initializer()

    #design a CNN
    #CNN layer 1 parameter
    L1_window_size = 6
    L1_depth = 1
    L1_num_kernel = 64
    L1_kernel_stride = []
    L1_pooling_stride = []
    K1_name = 'K1'
    b1_name = 'b1'
    #layer 1: convolution and max pooling
    with tf.variable_scope('C_Layer1') as scope:
        L1 = C_Layer(X_img, L1_window_size, L1_depth, L1_num_kernel, initializer, K1_name, b1_name, keep_prob)

    #CNN layer 2 parameter
    L2_window_size = 3
    L2_depth = L1_num_kernel
    L2_num_kernel = 128
    L2_kernel_stride = []
    L2_pooling_stride = []
    K2_name = 'K2'
    b2_name = 'b2'
    #layer 2: convolution and max pooling
    with tf.variable_scope('C_Layer2') as scope:
        L2 = C_Layer(L1, L2_window_size, L2_depth, L2_num_kernel, initializer, K2_name, b2_name, keep_prob)

    #CNN layer 3 parameter
    L3_window_size = 3
    L3_depth = L2_num_kernel
    L3_num_kernel = 256
    L3_kernel_stride = []
    L3_pooling_stride = []
    K3_name = 'K3'
    b3_name = 'b3'
    #layer 3: convolution and max pooling
    with tf.variable_scope('C_Layer3') as scope:
        L3 = C_Layer(L2, L3_window_size, L3_depth, L3_num_kernel, initializer, K3_name, b3_name, keep_prob)

    #CNN layer 4 parameter
    L4_window_size = 3
    L4_depth = L3_num_kernel
    L4_num_kernel = 512
    L4_kernel_stride = []
    L4_pooling_stride = []
    K4_name = 'K4'
    b4_name = 'b4'
    #layer 4: convolution and max pooling
    with tf.variable_scope('C_Layer4') as scope:
        L4 = C_Layer(L3, L4_window_size, L4_depth, L4_num_kernel, initializer, K4_name, b4_name, keep_prob)

    #reshape the final layer result to be fit to the following fully connected network (4D -> 2D)
    num_batch, final_cnn_height, final_cnn_width, final_cnn_depth = L4.shape
    L4 = tf.reshape(L4, [-1, final_cnn_width*final_cnn_height*final_cnn_depth])

    #design a fully connected network
    #FCN layer 1 parameter
    L1_dim = 1000
    W1_name = 'W1'
    b1_name = 'b1'
    #layer 1
    with tf.variable_scope('F_Layer1') as scope:
        FL1, W1, b1 = F_Layer(L4, L1_dim, initializer, W1_name, b1_name, keep_prob)

    #FCN layer 2 parameter
    L2_dim = 1000
    W2_name = 'W2'
    b2_name = 'b2'
    #layer 1
    with tf.variable_scope('F_Layer2') as scope:
        FL2, W2, b2 = F_Layer(FL1, L2_dim, initializer, W2_name, b2_name, keep_prob)

    #FCN layer 3 parameter
    L3_dim = 1000
    W3_name = 'W1'
    b3_name = 'b1'
    #layer 3
    with tf.variable_scope('F_Layer3') as scope:
        FL3, W3, b3 = F_Layer(FL2, L3_dim, initializer, W3_name, b3_name, keep_prob)

    #output layer
    with tf.variable_scope('Output') as scope:
        W4 = tf.get_variable("W4", shape=[L3_dim, num_class], initializer=initializer)
        b4 = tf.Variable(tf.random_normal([num_class]))

        #logit
        logit = tf.matmul(FL3, W4) + b4

    with tf.variable_scope('Training') as scope:
        #set the expoentially decaying learing rate

        hypothesis = tf.nn.softmax(logit, name = 'hypothesis')

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, when_to_decay, decay_ratio)

        #define cost function
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit, labels = Y_one_hot))

        #let the selected optimizer minimize the defined cost
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        #define the prediction accuracy
        correct_predition = tf.equal(tf.argmax(tf.nn.softmax(logit), 1), tf.argmax(Y_one_hot,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        print('Training Started')

        for epoch in range(num_epoch):

            num_batch = np.ceil(num_data / batch_size).astype(np.int16)
            avg_cost = 0
            avg_accu = 0

            for idx in range(num_batch):

                if idx == num_batch - 1:
                    batch_train_img = train_x_data[idx*batch_size:]
                    batch_train_label = train_y_data[idx*batch_size:]
                else:
                    batch_train_img = train_x_data[idx*batch_size:(idx+1)*batch_size]
                    batch_train_label = train_y_data[idx*batch_size:(idx+1)*batch_size]

                feed_dict = {X : batch_train_img, Y : batch_train_label, keep_prob : train_keep}
                acc, c, _ = sess.run([accuracy, cost, optimizer], feed_dict = feed_dict)
                avg_cost += c / num_batch
                avg_accu += acc/ num_batch
                #print('Epoch: ', '%04d' % (epoch+1), 'Batch idx: ', '%s' %(str(idx) + '/' +str(num_batch)), 'batch cost = ', '{:.9f}'.format(c), 'avg cost = ', '{:.9f}'.format(avg_cost), 'batch accu = ', '{:.9f}'.format(acc), 'avg accu = ', '{:.9f}'.format(avg_accu))

            print('Epoch: ', '%04d' % (epoch+1), 'Batch Size: ', batch_size, 'Initial Learning Rate: ', initial_learning_rate, 'Accuracy: ', avg_accu, 'Cost: ', avg_cost)


        num_batch = np.ceil(num_test / batch_size).astype(np.int16)
        avg_accu = 0

        for idx in range(num_batch):

            if idx == num_batch - 1:
                batch_test_img = test_x_data[idx*batch_size:]
                batch_test_label = test_y_data[idx*batch_size:]
            else:
                batch_test_img = test_x_data[idx*batch_size:(idx+1)*batch_size]
                batch_test_label = test_y_data[idx*batch_size:(idx+1)*batch_size]

            feed_dict = {X : batch_test_img, Y : batch_test_label, keep_prob : test_keep}

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

    accuracies, EER = CNN(data_set, labels, num_epoch, batch_size, initial_learning_rate, when_to_decay, decay_ratio, train_keep, test_keep)
    print(accuracies, EER)
    test_result.append(accuracies)

pickle.dump(test_result, open('data/classification_result.pkl', 'wb'))










































#end
