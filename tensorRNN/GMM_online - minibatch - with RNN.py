import numpy as np
import os
import sys
import tensorflow as tf
import warnings

#sys.path.append("E:\")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)
# np.random.seed(12345)
# tf.random.set_random_seed(12345)

from mytools import load_model
from mytools import logsumexp
# from mytools import sigmoid
# from mytools import softmax
from collections import Counter
from tRNN_UVW_tf import MyRnn
from scipy.special import digamma

def logsumexp(*args, **kwargs):
    import scipy
    return scipy.special.logsumexp(*args, **kwargs)

def load_model(file_name):
    import pickle
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model

def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))
    # inputs = np.asarray(inputs)
    #
    # input_shape = inputs.shape
    # inputs = np.reshape(inputs, [1, -1])
    # result = [1 / float(1 + np.exp(-x)) for x in inputs[0]]  # 这个0指第0行，一共就0行
    # return np.reshape(np.asarray(result), input_shape)


def softmax(inputs, axis=0):
    tmp = np.exp(inputs)
    return tmp / np.expand_dims(np.sum(tmp, axis=axis), axis)

def load_data(win_len, overlap, maxT):
    from mytools import loadData
    traindata, trainlabel, testdata, testlabel = loadData(win_len, overlap, maxT)
    return traindata, trainlabel, testdata, testlabel


def get_Rnk(X, GMM_param, RNN_y):

    means, covars, log_P, log_lamda, beta_lamda_inv = GMM_param
    log_pnk = -0.5 * (inputSize * np.log(2 * np.pi)
                      - np.sum(log_lamda, 1)
                      + np.sum((np.power(means, 2) + beta_lamda_inv) / covars, 1)
                      - 2 * np.matmul(X, np.transpose(means / covars))
                      + np.matmul(np.power(X, 2), np.transpose((1.0 / covars)))) + log_P + np.log(RNN_y)   # 这里用到RNN_Y是RNNprediction的概率 我们改成我们自己的loss得到的值就行 用回归的

    log_sumPnk = logsumexp(log_pnk, 1)
    responsibility = np.exp(log_pnk - np.expand_dims(log_sumPnk, 1))  # Z的后验 b*k
    return responsibility


def get_N_k(z):
    return np.sum(z, 0)  # z : N*k


def get_xHat_k(x, z, N_k):
    K = z.shape[1]
    xHat = []
    for k in range(K):
        if np.abs(N_k[k]) < 0.0001:
            xHat.append(np.zeros(inputSize))
        else:
            temp = np.sum(np.reshape(z[:, k], [-1, 1]) * x, 0)
            xHat.append(temp / N_k[k])
    # print(xHat, N_k )
    return np.asarray(xHat)


def get_sumN_Znk_mul_squareXn(x, z):
    K = z.shape[1]
    V_K = []
    for k in range(K):
        V_K.append(np.sum(np.reshape(z[:, k], [-1, 1]) * x ** 2, 0))
    return np.asarray(V_K)


def _get_sum_back_N_k(N_k):
    '''返回后向累加'''
    res = np.zeros(len(N_k))
    for i in range(len(N_k) - 1 - 1, -1, -1):
        res[i] = N_k[i + 1] + res[i + 1]
    return res


def update(N_k, xHat_k, V_k, param_old, rho, times):
    # 参数更新
    # alpha_K = alpha_0 + N_k * times
    alpha_sb0_K = 1 + N_k * times
    alpha_sb1_K = alpha_dp + _get_sum_back_N_k(N_k) * times
    beta_K = beta_0 + N_k * times
    c_K = c_0 + 0.5 * N_k * times

    m_K, d_K = m_0 * 0, d_0 * 0
    for k in range(MAX_cluster_K):
        m_K[k] = (xHat_k[k] * N_k[k] * times + beta_0[k] * m_0[k]) / beta_K[k]
        # d_K = 0.5 * (2 * d_0 + N_k * S_k + (beta_0 * N_k) / (beta_0 + N_k) * (xHat_k - m_0) ** 2)
        d_K[k] = 0.5 * (
                V_k[k] * times +
                beta_0[k] * m_0[k] ** 2 +
                2 * d_0[k] -
                beta_K[k] * m_K[k] ** 2)
    # assert d_K == d_K2
    # print(m_K)
    param_temp = (alpha_sb0_K, alpha_sb1_K, beta_K, m_K, c_K, d_K)  #根据minibatch更新参数，得到temp参数，更新全局参数
    param_new = [(1 - rho) * aa + rho * bb for aa, bb in zip(param_old, param_temp)]
    return param_new


def get_log_P_use_stickBreaking(alpha_sb0_0, alpha_sb1_0):   #得到z的先验分布，DP

    Exp_log_vk = digamma(alpha_sb0_0) - digamma(alpha_sb0_0 + alpha_sb1_0)
    Exp_log_1_vi = digamma(alpha_sb1_0) - digamma(alpha_sb0_0 + alpha_sb1_0)
    sum_Exp_log_1_vi = np.cumsum(Exp_log_1_vi)  # 计算log(1-vi)的累加和
    Exp_log_vk[-1] = 0  # v最后一段假设取所有，那么它的期望就是100%-->1 取对数就是0

    log_P = np.zeros(MAX_cluster_K)
    log_P[0] = Exp_log_vk[0]
    for i in range(1, MAX_cluster_K):
        log_P[i] = sum_Exp_log_1_vi[i - 1] + Exp_log_vk[i]
    return log_P
    # v = alpha_sb0_0 / (alpha_sb0_0 + alpha_sb1_0)
    # v[-1] = 1  # 最后一段占据所有剩余长度
    # P = np.zeros(MAX_cluster_K)
    # stick = 1
    # for i in range(MAX_cluster_K):
    #     P[i] = stick * v[i]
    #     stick *= (1 - v[i])
    # return P


# 使用期望计算初始参数，初始化模型（我们本身也是基于mean-VB更新的）
def get_new_param(alpha_sb0_0, alpha_sb1_0, m_0, beta_0, c_0, d_0):
    # P = alpha_0 / np.sum(alpha_0)  # π
    log_P = get_log_P_use_stickBreaking(alpha_sb0_0, alpha_sb1_0)
    lamda = np.reshape(c_0, [-1, 1]) / d_0  # λ
    means = m_0  # μ

    log_lamda = digamma(np.reshape(c_0, [-1, 1])) - np.log(d_0)

    covars = lamda ** -1
    beta_lamda_inv = (np.reshape(beta_0, [-1, 1]) ** -1) * covars
    GMM_param = [means, covars, log_P, log_lamda, beta_lamda_inv]
    return GMM_param


# def get_Y_useRNN_np(batch_x, hid_old, RNN_param, lastTime_cluster=-1):
#     '''
#     :param batch_x: batch*32
#     :param hid_old: batch*hid
#     :return: Y batch*3
#     '''
#     U, V, W = RNN_param
#     '''hid_new (20, 2, 20) '''
#     if WHH_mode == "back":
#         hid_new = sigmoid(np.tensordot(batch_x, U, [[1], [1]]) +
#                           np.tensordot(hid_old, W, [[1], [1]]))
#     elif WHH_mode == "same":
#         hid_new = sigmoid(np.tensordot(batch_x, U, [[1], [1]]) +
#                           np.expand_dims(np.matmul(hid_old, W[0]), 1))  # 用第0个
#     elif WHH_mode == "for":
#         W2 = get_new_WHH(lastTime_cluster, W)
#         # W2 : n*hid*hid
#         hid_new = sigmoid(np.tensordot(batch_x, U, [[1], [1]]) +
#                           np.matmul(np.expand_dims(hid_old, 1), W2))
#
#     hid_new2 = np.transpose(hid_new, [1, 0, 2])
#     output = softmax(np.matmul(hid_new2, V), 2)
#     output = np.transpose(output, [1, 0, 2])
#
#     '''hid_new (20, 2, 20), output(20, 2, 3)'''
#     return hid_new, output


# def get_new_WHH(cluster, W_tensor):
#     ret = []
#     for i in range(len(cluster)):
#         ret.append(W_tensor[int(cluster[i])])
#     return np.asarray(ret)
#
#
# def get_new_hid(hid_old, responsibility):
#     responsibility = np.argmax(responsibility, 1)
#     res = []
#     miniBatch_Size = len(responsibility)
#     for i in range(miniBatch_Size):
#         res.append(hid_old[i][responsibility[i]])
#     res = np.asarray(res)
#     return res


def get_responsibility(batch_x, batch_y, GMM_param, RNN_param, mode='train'):

    if mode == 'train':
        responsibility = []

        #maxTime = batch_x.shape[1]
        miniBatch_Size = batch_x.shape[0]
        hid_old = np.zeros([miniBatch_Size, hiddenSize])



        for t in range(maxTime):
            batch_x_temp = batch_x[:, t]

            if WHH_mode == "back" or WHH_mode == "same":

                rnn_hid, RNN_y = get_Y_useRNN_np(batch_x_temp, hid_old, RNN_param)
            elif WHH_mode == "for":

                if t == 0:
                    rnn_hid, RNN_y = get_Y_useRNN_np(batch_x_temp, hid_old, RNN_param,
                                                     lastTime_cluster=np.zeros(
                                                         [miniBatch_Size]))
                else:
                    rnn_hid, RNN_y = get_Y_useRNN_np(batch_x_temp, hid_old, RNN_param,
                                                     lastTime_cluster=np.argmax(resp_temp, 1))
            else:
                raise PermissionError

        RNN_y = np.max(np.expand_dims(batch_y, 1) * RNN_y, 2)  # b*k


        resp_temp = get_Rnk(np.reshape(batch_x, [-1, inputSize]), GMM_param, RNN_y=RNN_y)
        responsibility.append(resp_temp)  # B,K

        # hid_old = get_new_hid(rnn_hid, resp_temp)

        responsibility = np.transpose(np.asarray(responsibility), [1, 0, 2])
        responsibility = np.reshape(responsibility, [miniBatch_Size, -1])
    elif mode == 'test':
        responsibility = get_Rnk(np.reshape(batch_x, [-1, inputSize]), GMM_param, RNN_y=1)
    else:
        raise Exception("no this mode")
    return responsibility


# def __get_acc_use_pred(pred, label):
#     label = np.argmax(label, 1)
#     n = len(label)
#     # vote:
#     pred1 = np.argmax(pred, 2)
#     tmp = np.zeros(n)
#     for i in range(n):
#         dic = Counter(pred1[:, i])
#         tmp[i] = max(dic)
#     acc_vote = np.average(np.equal(label, tmp))
#     # prod
#     pred3 = np.prod(pred, 0)
#     # pred3 = np.sum(pred, 0)
#     pred4 = np.argmax(pred3, 1)
#     acc_prod = np.average(np.equal(label, pred4))
#     return acc_vote, acc_prod
#
#
# def get_acc_use_pred(pred, label, mode="ALL"):
#     if mode == "ALL":
#         return __get_acc_use_pred(pred, label)

#     # pred (15, 7800, 3)
#     # label (7800, 3)
#

#     acc_vote = []
#     acc_prod = []
#     for i in range(timeStep):
#         temp1, temp2 = __get_acc_use_pred(pred[:i + 1], label)
#         acc_vote.append(temp1)
#         acc_prod.append(temp2)
#     return np.asarray(acc_vote).reshape([1, -1]), np.asarray(acc_prod).reshape([1, -1])


class Main():
    def __init__(self):

        self.alpha_sb0_K = alpha_sb0_0
        self.alpha_sb1_K = alpha_sb1_0
        self.m_K = m_0
        self.beta_K = beta_0
        self.c_K = c_0
        self.d_K = d_0

        self.l_r_init = lr_init
        self.batchSizeTrain = miniBatch_Size
        self.hiddenSize = hiddenSize
        self.maxEpoch = maxEpoch

    def GMM_main(self, batch_x, batch_y, GMM_param, epoch, times, RNN_param):

        # GMM_param = [means, covars, log_P, log_lamda,beta_lamda_inv]
        responsibility = get_responsibility(batch_x, batch_y, GMM_param, RNN_param)


        batch_x = np.reshape(batch_x, [-1, inputSize])

        N_K = get_N_k(responsibility)  # K
        xHat_K = get_xHat_k(batch_x, responsibility, N_K)  # K*P
        V_K = get_sumN_Znk_mul_squareXn(batch_x, responsibility)

        rho = (tau + epoch) ** -Kappa
        param = (self.alpha_sb0_K, self.alpha_sb1_K, self.beta_K, self.m_K, self.c_K, self.d_K)
        self.alpha_sb0_K, self.alpha_sb1_K, self.beta_K, self.m_K, self.c_K, self.d_K = update(N_K, xHat_K, V_K, param,
                                                                                               rho,
                                                                                               times=times)

        GMM_param = get_new_param(self.alpha_sb0_K, self.alpha_sb1_K, self.m_K, self.beta_K, self.c_K, self.d_K) #alpha_sb0_K，alpha_sb1_K分别对应推导公式中的beta分布的两个参数，stick-breaking
        # covars = lamda ** -1
        return GMM_param, responsibility

    # def get_UVW_np(self, sess, RNN_model):
    #     RNN_param = sess.run([RNN_model.U, RNN_model.V, RNN_model.W])
    #     return RNN_param

    def run(self):

        traindataSize = traindata.shape[0]  # 1500*32

        GMM_param = get_new_param(alpha_sb0_0, alpha_sb1_0, m_0, beta_0, c_0, d_0)

        # rnn = MyRnn(l_r_init=lr_init,
        #             batchSizeTrain=miniBatch_Size,
        #             inputSize=inputSize,
        #             hiddenSize=hiddenSize,
        #             outputSize=outputSize,
        #             timeStep=timeStep,
        #             cluster_K=MAX_cluster_K,
        #             alpha_regularizer=alpha_regularizer,
        #             WHH_mode=WHH_mode
        #             )
        # optimizer, pred, Loss, hiddens = rnn.get_opt_acc()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_USED)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # sess.run(tf.global_variables_initializer())

        self.trainAcc_record = []
        self.testAcc_record = []
        self.cluster_resulr_tr = []
        self.cluster_resulr_te = []
        for epoch in range(maxEpoch):
            for j in range(int(traindataSize / miniBatch_Size)):
                batch_x = traindata[miniBatch_Size * j:miniBatch_Size * (j + 1)]
                batch_y = trainlabel[miniBatch_Size * j:miniBatch_Size * (j + 1)]


                RNN_param = self.get_UVW_np(sess, rnn)
                new_GMM_param, responsibility = self.GMM_main(batch_x, batch_y, GMM_param,
                                                              epoch * 1,
                                                              int(
                                                                  traindataSize / miniBatch_Size),
                                                              RNN_param)     ##这里的RNN——param就是我们model的parameter，为了得到regress，做后验的传递
                '''
                # GMM_param = [means, covars, log_P, log_lamda,beta_lamda_inv]
                # means, covars, log_P, log_lamda,beta_lamda_inv = GMM_param
                np.exp(log_P)
                np.sum(np.exp(log_P))
                means.shape
                covars.shape
                '''
                GMM_param = new_GMM_param
                lr_temp = rnn.l_r_init / (epoch * 0.1 + 1)

                batch_clusterLabel = np.reshape(np.argmax(responsibility, 1), [-1, timeStep])
                sess.run(optimizer, feed_dict={
                    rnn.x: batch_x,
                    rnn.y: batch_y,
                    rnn.l_r: lr_temp,
                    rnn.cluster_label: batch_clusterLabel
                })

            RNN_param = self.get_UVW_np(sess, rnn)
            responsibility = get_responsibility(traindata, trainlabel, GMM_param, RNN_param, mode='train')
            batch_clusterLabel = np.reshape(np.argmax(responsibility, 1), [-1, timeStep])
            self.cluster_resulr_tr.append(batch_clusterLabel)
            pred1 = sess.run(pred, feed_dict={
                rnn.x: traindata,
                rnn.y: trainlabel,
                rnn.cluster_label: batch_clusterLabel
            })
            responsibility = get_responsibility(testdata, testlabel, GMM_param, RNN_param, mode='test')
            batch_clusterLabel = np.reshape(np.argmax(responsibility, 1), [-1, timeStep])
            self.cluster_resulr_te.append(batch_clusterLabel)
            pred2, loss = sess.run([pred, Loss], feed_dict={
                rnn.x: testdata,
                rnn.y: testlabel,
                rnn.cluster_label: batch_clusterLabel
            })
            acc1_vote, acc1_prod = get_acc_use_pred(pred1, trainlabel)
            acc2_vote, acc2_prod = get_acc_use_pred(pred2, testlabel)
            self.trainAcc_record.append(max(acc1_vote, acc1_prod))
            self.testAcc_record.append(max(acc2_vote, acc2_prod))
            # 计算当前聚类个数
            current_cluster_count = len((set((list(np.reshape(batch_clusterLabel, [1, -1]))[0]))))
            if epoch % 1 == 0:
                print("epoch: %d  trAc: %.4f  teAc: %.4f || loss: %.2f  cluster: %d"
                      % (epoch, max(acc1_vote, acc1_prod), max(acc2_vote, acc2_prod), loss, current_cluster_count))

        # def _save_func():
        #     param_dict = {}
        #     RNN_param = self.get_UVW_np(sess, rnn)
        #     responsibility1 = get_responsibility(traindata, trainlabel, GMM_param, RNN_param, mode='train')
        #     responsibility2 = get_responsibility(testdata, testlabel, GMM_param, RNN_param, mode='test')
        #     batch_clusterLabel1 = np.reshape(np.argmax(responsibility1, 1), [-1, timeStep])
        #
        #     pred1, hiddens1 = sess.run([pred, hiddens], feed_dict={
        #         rnn.x: traindata,
        #         rnn.y: trainlabel,
        #         rnn.cluster_label: batch_clusterLabel1})
        #     acc1_vote, acc1_prod = get_acc_use_pred(pred1, trainlabel, "timechange")  # 不同时刻的识别性能
        #     param_dict["hid_tr"] = hiddens1
        #     param_dict["acc_allTime_tr"] = np.max([acc1_vote, acc1_prod], 0)
        #     return param_dict

        # def _save_func2():
        #     param_dict = {}
        #     param_dict["cluster_resulr_tr"] = self.cluster_resulr_tr[-1]
        #     param_dict["cluster_resulr_te"] = self.cluster_resulr_te[-1]
        #     return param_dict

        # self.param_dict
        # self.param_dict = _save_func()
        # self.param_dict = {}
        self.param_dict = _save_func2()
        print("finish")


from mytools import save_model_to_mat
from mytools import RNN_save_model_v4 as RNN_save_model  # 有存储字典
# from mytools import RNN_save_model_v0 as RNN_save_model #无存储字典
from mytools import check_folder
from mytools import save_model
from mytools import draw


def start():

    myRNN = []
    for _ in range(5):
        rnn = Main()
        rnn.run()

        myRNN.append(RNN_save_model(rnn))

    fileName = ("two/test_%d_" % (runTime))

    check_folder(fileName.split('/')[0])
    # save_model(fileName + ".pkl", myRNN)
    draw(fileName + "_", myRNN)
    save_model_to_mat(fileName + "_", myRNN)


if __name__ != '__main__':
    raise PermissionError

for runTime in range(20):
    # for win_len in [52,56]:

    # win_len = 32
    # overlap = int(0.5 * win_len)
    #
    # timeStep = int(((256 - overlap) / (win_len - overlap)))
    # inputSize = win_len
    # outputSize = 3
    traindata, trainlabel, testdata, testlabel = load_data(win_len, overlap, timeStep)

    '''DP'''
    alpha_dp = 1
    '''GMM'''
    Kappa = 0.8  # (0.5,1]
    tau = 1000
    MAX_cluster_K = 20
    '''RNN'''
    lr_init = 0.017
    hiddenSize = 20
    miniBatch_Size = 20
    alpha_regularizer = 0.0003 * 1
    maxEpoch = 5
    GPU_USED = 1.00
    WHH_mode = "for"  # for back same

    alpha_sb0_0 = np.ones([MAX_cluster_K])
    alpha_sb1_0 = np.ones([MAX_cluster_K])
    beta_0 = np.ones([MAX_cluster_K])
    c_0 = np.ones([MAX_cluster_K])
    m_0 = np.ones([MAX_cluster_K, inputSize])
    d_0 = np.ones([MAX_cluster_K, inputSize])

    #initializeing with GMM

    model = load_model(r"E:\tensorRNN\cluster_gmm\estimator_K_" + str(MAX_cluster_K) + ".pkl")
    temp = np.mean(model.means_, 1)
    for i in range(MAX_cluster_K):
        m_0[i] *= temp[i]


    start()

    print(" end this way ")
