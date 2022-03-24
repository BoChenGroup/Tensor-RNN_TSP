import os
import sys

import tensorflow as tf

sys.path.append("E:\our python")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


# tf.set_random_seed(12345)

# timeStep = 15
# inputSize = 32
# outputSize = 3  # =n_classes


class MyRnn():
    def __init__(self, l_r_init, batchSizeTrain, inputSize, hiddenSize, outputSize, timeStep, cluster_K,
                 alpha_regularizer, WHH_mode):
        self.l_r_init = l_r_init
        self.batchSizeTrain = batchSizeTrain
        self.hiddenSize = hiddenSize
        self.cluster_K = cluster_K
        self.WHH_mode = WHH_mode
        self.inputSize = inputSize
        self.timeStep = timeStep

        # RNN 用到的参数
        self.U = [tf.Variable(tf.random_normal([inputSize, hiddenSize])) for i in range(cluster_K)]
        self.V = [tf.Variable(tf.random_normal([hiddenSize, outputSize])) for i in range(cluster_K)]
        self.W = [tf.Variable(tf.random_normal([hiddenSize, hiddenSize])) for i in range(cluster_K)]
        self.alpha_regularizer = alpha_regularizer

        self.x = tf.placeholder("float", [None, timeStep, inputSize])
        self.y = tf.placeholder("float", [None, outputSize])
        self.cluster_label = tf.placeholder("int32", [None, timeStep])
        self.l_r = tf.placeholder("float")

    def RNN_train(self, x, cluster_label):
        def rnn_cell(input_x, state, W_temp, U_temp):
            state = tf.expand_dims(state, 1)
            input_x = tf.expand_dims(input_x, 1)
            state = tf.sigmoid(tf.squeeze(tf.matmul(input_x, U_temp), 1) +
                               tf.squeeze(tf.matmul(state, W_temp), 1))
            return state

        x = tf.transpose(x, [1, 0, 2])  # x :(15, ?, 32)
        x = tf.reshape(x, [-1, self.inputSize])  # shape=(?, 32)
        x = tf.split(x, self.timeStep, 0)  # 15 split ，split：(15,?, 32)

        state = tf.matmul(x[0], self.U[0]) * 0  # 维度初始化
        hiddens =[]
        outputs = []
        for ii in range(len(x)):
            input_X = x[ii]
            input_cluster_label = cluster_label[:, ii]

            if self.WHH_mode == "for":

                if ii > 0:
                    W_temp = tf.gather(self.W, cluster_label[:, ii - 1])  # Whh
                else:
                    W_temp = tf.gather(self.W, input_cluster_label)  # Whh i=0时 这一项可以随便取，因为hid=0
            elif self.WHH_mode == "back":

                W_temp = tf.gather(self.W, input_cluster_label)  # Whh
            elif self.WHH_mode == "same":

                W_temp = tf.gather(self.W, input_cluster_label * 0)  # Whh
            else:
                raise PermissionError


            U_temp = tf.gather(self.U, input_cluster_label)  # Whx

            # U_temp = tf.gather(self.U, input_cluster_label * 0)  # Whx


            V_temp = tf.gather(self.V, input_cluster_label)  # Why

            # V_temp = tf.gather(self.V, input_cluster_label*0)  # Why

            state = rnn_cell(input_X, state, W_temp, U_temp)
            hiddens.append(state)


            state = tf.expand_dims(state, 1)
            outputs.append(tf.squeeze(tf.matmul(state, V_temp), 1))
            state = tf.squeeze(state, 1)

        outputs = tf.nn.softmax(outputs, axis=2)
        hiddens = tf.convert_to_tensor(hiddens)
        # self.output_temp = outputs
        return outputs, hiddens  # (15,?,3),(15,?, 20)

    def _L2_regularizer(self, cost):
        if self.alpha_regularizer > 0:
            for i in range(self.cluster_K):
                cost += tf.contrib.layers.l2_regularizer(self.alpha_regularizer)(self.U[i])
                cost += tf.contrib.layers.l2_regularizer(self.alpha_regularizer)(self.V[i])
                cost += tf.contrib.layers.l2_regularizer(self.alpha_regularizer)(self.W[i])
        return cost

    def _get_param_dict(self, sess, paramTest):

        U2 = tf.convert_to_tensor(self.U)
        V2 = tf.convert_to_tensor(self.V)
        W2 = tf.convert_to_tensor(self.W)
        param_value = sess.run([U2, V2, W2, self.output_temp, self.cluster_label, self.y], feed_dict={
            self.x: paramTest[0],
            self.y: paramTest[1],
            self.cluster_label: paramTest[2]})
        dict_key = ['U', 'V', 'W', 'output_val', 'cluster_label', 'label']
        param_dict = dict(zip(dict_key, param_value))
        return param_dict

    def get_opt_acc(self):
        pred, hiddens = self.RNN_train(self.x, self.cluster_label)
        # pred_prod = tf.reduce_prod(input_tensor=pred, axis=0)  # (15,20,3)
        pred_temp = tf.split(pred, self.timeStep, 0)
        cost = 0
        for pred_i in pred_temp:
            cost += -tf.reduce_sum(self.y * tf.log(pred_i))
        cost /= len(pred_temp)
        cost = self._L2_regularizer(cost=cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.l_r).minimize(cost)
        # correct_pred = tf.equal(tf.argmax(pred_prod, 1), tf.argmax(self.y, 1))

        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print("RNN INIT FINISH")
        return optimizer, pred, cost, hiddens
