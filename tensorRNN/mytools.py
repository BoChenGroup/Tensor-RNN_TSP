import numpy as np

print("load mytools")


def logsumexp(*args, **kwargs):
    import scipy
    return scipy.special.logsumexp(*args, **kwargs)


def order_data(data, label):
    K = np.shape(label)[1]

    label = np.argmax(label, 1)
    data_2 = []
    label_2 = []
    for i in range(K):
        data_2.append(data[label == i])
        label_2.append(np.tile(np.eye(K)[i], [data_2[-1].shape[0], 1]))
    return np.concatenate(data_2), np.concatenate(label_2)


def rand_order(arg):
    return np.random.permutation(arg)


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


def check_folder(folder_path):
    import os
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def save_model(file_name, model):
    import pickle
    with open(file_name, 'wb+') as f:
        pickle.dump(model, f)


def load_model(file_name):
    import pickle
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model


def save_model_to_mat(fileName, model):

    for i in range(len(model)):
        param_dict = model[i].param_dict

        param_dict['trainResult'] = model[i].trainResult
        param_dict['testResult'] = model[i].testResult
        save_data_to_mat(param_dict, fileName=fileName + str(i) + '.mat')


def save_data_to_mat(input_dic, fileName="temp.mat"):

    import scipy.io as sio
    sio.savemat(fileName, input_dic)



def draw(file_name, model, select=1):
    import matplotlib.pyplot as plt
    import numpy as np
    if select == 2:
        file_name += '[2]'
    for i in range(len(model)):
        if select == 1:
            y1 = model[i].trainResult
            y2 = model[i].testResult
        elif select == 2:
            y1 = model[i].trainResult2
            y2 = model[i].testResult2
        x = range(len(y1))
        plt.clf()
        plt.plot(x, y1, 'r-', x, y2, 'g-')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(['train data', 'test data'])
        title = ("%.4f    %.4f    %.4f" % (max(y1), max(y2), np.average(y2[-15:])))
        plt.title(title)
        plt.savefig(file_name + str(i) + "_" + str(round(np.average(y2[-15:]), 4)) + ".png", bbox_inches='tight')


draw_v0 = draw



def draw_v2(file_name, model):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(len(model)):
        y1 = model[i].trainResult
        y2 = model[i].testResult
        y3 = model[i].testResult_for
        y4 = model[i].testResult_bac
        x = range(len(y1))
        plt.clf()
        plt.plot(x, y1, 'r-', x, y2, 'g-', x, y3, 'y-', x, y4, 'c-')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(['train data', 'test data', 'test data for', 'test data bac'])
        title = ("%.4f    %.4f    %.4f" % (max(y1), max(y2), np.average(y2[-5:])))
        plt.title(title)
        plt.savefig(file_name + str(i) + ".png")


def save_param_to_txt_v0(file_name, model, changed_param=""):

    if len(model) == 0:
        return
    with open(file_name, 'w') as f:
        f.write("Param Setting: \n")
        f.write("    l_r_init  :\t" + str(model[0].l_r_init) + "\n")
        f.write("    batchSize :\t" + str(model[0].batchSizeTrain) + "\n")
        f.write("    hiddenSize:\t" + str(model[0].hiddenSize) + "\n")
        f.write("    maxEpoch  :\t" + str(model[0].maxEpoch) + "\n")
        f.write("\n")

        f.write("    **changed_param** :\t" + str(changed_param) + "\n")
        f.write("\n\n")

        for i in range(len(model)):
            f.write("number " + str(i) + "\n")
            y1 = model[i].trainResult
            y2 = model[i].testResult
            f.write("    max_train_result:\t" + str(round(max(y1), 4)) + "\n")
            f.write("    max_test_result:\t" + str(round(max(y2), 4)) + "\n")
            f.write("    avg_train_result:\t" + str(round(np.average(y1[-10:]), 4)) + "\n")
            f.write("    avg_test_result:\t" + str(round(np.average(y2[-10:]), 4)) + "\n")
            f.write("\n")


def save_param_to_txt(file_name, model, changed_param=""):

    if len(model) == 0:
        return
    with open(file_name, 'w') as f:
        f.write("Param Setting: \n")
        f.write("    l_r_init  :\t" + str(model[0].l_r_init) + "\n")
        f.write("    batchSize :\t" + str(model[0].batchSizeTrain) + "\n")
        f.write("    hiddenSize:\t" + str(model[0].hiddenSize) + "\n")
        f.write("    maxEpoch  :\t" + str(model[0].maxEpoch) + "\n")
        f.write("    cluster_K :\t" + str(model[0].cluster_K) + "\n")  # add
        f.write("    alpha_regul:\t" + str(model[0].alpha_regularizer) + "\n")  # add
        f.write("\n")

        f.write("    **changed_param** :\t" + str(changed_param) + "\n")
        f.write("\n\n")

        for i in range(len(model)):
            f.write("number " + str(i) + "\n")
            y1 = model[i].trainResult
            y2 = model[i].testResult
            f.write("    max_train_result:\t" + str(round(max(y1), 4)) + "\n")
            f.write("    max_test_result:\t" + str(round(max(y2), 4)) + "\n")
            f.write("    avg_train_result:\t" + str(round(np.average(y1[-10:]), 4)) + "\n")
            f.write("    avg_test_result:\t" + str(round(np.average(y2[-10:]), 4)) + "\n")
            f.write("\n")


def my_svm(X, Y):
    # from sklearn.svm import SVC
    # clf = svm.SVC(decision_function_shape='ovo')
    # clf.fit(X, Y)
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)
    pass


def PCA_2D(x, y, file_name):
    # input: (7800*15)*32, 7800*3
    import numpy as np
    from matplotlib import cm
    import pylab as pl
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca_x = pca.fit_transform(x)
    ymin = np.min(y)
    ymax = np.max(y)
    #    line_color = ['ob','og','or','oc','om','oy','ok','*b','*g','*r','*c','*m','*y','*k']
    cm_subsection = np.linspace(0.0, 1.0, ymax - ymin + 1)
    colors = [cm.jet(k) for k in cm_subsection]
    pl.figure(1)
    for i in range(y.shape[1]):  # 3
        indx = np.where(y == i)
        x1 = []
        x2 = []
        for j in range(len(indx)):
            m = indx[j]
            x1.append(pca_x[m][0])
            x2.append(pca_x[m][1])
        pl.scatter(x1, x2, color=colors[i], marker='o', label=u"class" + str(i), s=5)

        pl.title(u"PCA_2D_Projection")
        pl.xlabel(u'Dim_0')
        pl.ylabel(u'Dim_1')
        pl.legend()
    pl.savefig(file_name)


def draw_a_matrix(input, fileName="temp.png"):

    import numpy as np
    import matplotlib.pyplot as plt
    y = np.reshape(input, [-1, 1])
    x = range(len(y))
    plt.clf()
    plt.plot(x, y, 'r-')
    plt.savefig(fileName)
    plt.clf()
    plt.close()



def draw_a_matrix_v2(b, fileName="temp.png"):
    import matplotlib.pyplot as plt
    plt.clf()
    xx, yy = b.shape
    plt.figure(figsize=(xx, yy))

    plt.imshow(b, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(True, which='minor', linestyle='--', linewidth="1")

    plt.gca().set_xticks(np.array(range(yy)) + 0.5, minor=True)
    plt.gca().set_yticks(np.array(range(xx)) + 0.5, minor=True)

    xlocations = np.array(range((yy)))
    plt.xticks(xlocations, xlocations + 1)
    ylocations = np.array(range((xx)))
    plt.yticks(ylocations, ylocations + 1)
    plt.savefig(fileName)
    plt.clf()
    plt.close()


def draw_predictRight_count(input):
    import numpy as np
    import matplotlib.pyplot as plt
    y = np.reshape(input, [-1, 1])
    x = range(len(y))
    plt.clf()
    plt.plot(x, y, 'r-')
    plt.savefig("temp.png")
    plt.clf()


def loadData(win_len=32, overlap=16, maxT=15):

    import scipy.io as sio
    # trainPath = r"E:\2018年9月14日 tensorRNN\0 其他工具\hrrp_train.mat"
    # testPath = r"E:\2018年9月14日 tensorRNN\0 其他工具\hrrp_test.mat"
    # 文件名格式 hrrp_test_32_24_29.mat
    trainPath = r"E:\tensorRNN\hrrp_train_"+str(win_len)+"_"+str(overlap)+"_"+str(maxT)+".mat"
    testPath = r"E:\tensorRNN\hrrp_test_"+str(win_len)+"_"+str(overlap)+"_"+str(maxT)+".mat"

    data = sio.loadmat(trainPath)
    traindata = data['traindata']  # (117000,32)
    # traindata = np.reshape(traindata, [-1, maxT, win_len])  # 7800*15*32
    selftraindata = traindata
    trainlabel = data['trainlabel']
    # trainlabel = trainlabel.T
    selftrainlabel = trainlabel

    data = sio.loadmat(testPath)
    testdata = data['testdata']  # (117000,32)
    # testdata = np.reshape(testdata, [-1, maxT, win_len])
    selftestdata = testdata
    testlabel = data['testlabel']
    # testlabel = testlabel.T
    selftestlabel = testlabel

    return selftraindata, selftrainlabel, selftestdata, selftestlabel


def loadData_xubin(winlen):

    import scipy.io as sio
    trainPath = r"C:\Users\ljq\Desktop\tensorRNN\hrrp_train winlen=" + str(
        winlen) + ".mat"
    testPath = r"C:\Users\ljq\Desktop\tensorRNN\hrrp_test winlen=" + str(
        winlen) + ".mat"

    data = sio.loadmat(trainPath)
    traindata = data['traindata']  # (117000,32)
    traindata = np.reshape(traindata, [7800, -1, winlen])  # 7800*15*32
    selftraindata = traindata
    trainlabel = data['trainlabel']
    trainlabel = np.reshape(trainlabel, [7800, -1, 3])
    selftrainlabel = trainlabel

    data = sio.loadmat(testPath)
    testdata = data['testdata']  # (117000,32)
    testdata = np.reshape(testdata, [5124, -1, winlen])
    selftestdata = testdata
    testlabel = data['testlabel']
    testlabel = np.reshape(testlabel, [5124, -1, 3])
    selftestlabel = testlabel

    return selftraindata, selftrainlabel, selftestdata, selftestlabel


def loadCluster(K, flag='kmeans'):
    if flag == 'kmeans':
        fileName = r"E:\2018年9月14日 tensorRNN\cluster_kmeans\cluster_K_" + str(K) + ".pkl"
    else:
        fileName = r"E:\2018年9月14日 tensorRNN\cluster_gmm\cluster_K_" + str(K) + ".pkl"

    [trainResult, trainDic, testResult, testDic] = load_model(file_name=fileName)
    trainResult = np.reshape(trainResult, [7800, 15])
    testResult = np.reshape(testResult, [5124, 15])
    return np.asarray([trainResult, trainDic, testResult, testDic])


def loadCluster_xubin(winlen):
    fileName = r"cluster_gmm\cluster_K_2_winlen_" + str(winlen) + ".pkl"

    [trainResult, trainDic, testResult, testDic] = load_model(file_name=fileName)
    trainResult = np.reshape(trainResult, [7800, -1])
    testResult = np.reshape(testResult, [5124, -1])
    return [trainResult, trainDic, testResult, testDic]


def __calcu_cluster_center(input_x, label, K):

    from collections import Counter

    num, T, inputSize = input_x.shape
    result = np.zeros([num, K, inputSize])
    for i in range(num):
        for t in range(T):
            result[i, label[i, t]] += input_x[i, t]

        cluster_count = dict(Counter(label[i]))
        for q, v in cluster_count.items():
            result[i, q] /= v

    return result
    # save_model('cluster_center_' + str(K) + '_.pkl', result)


def load_cluster_center(K):
    fileName = r"E:\tensorRNN\cluster_center_" + str(K) + ".pkl"
    [train_cluCenter, test_cluCenter] = load_model(file_name=fileName)
    return [train_cluCenter, test_cluCenter]


def get_run_time(func_run):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func_run(*args, **kwargs)
        end = time.time()
        print('time ：  %d s' % (end - start))
        return result

    return wrapper


def get_confMat(pred, label):
    import numpy as np
    sampleNum, classNum = label.shape
    label = np.argmax(label, 1)
    pred = np.argmax(pred, 1)

    result = np.zeros([classNum, classNum], dtype="int32")
    for index in range(sampleNum):
        row = label[index]
        col = pred[index]
        result[row, col] += 1

    return result


#######################################


#######################################

class RNN_save_model_v0():
    def __init__(self, rnn_Model):
        self.l_r_init = rnn_Model.l_r_init
        self.batchSizeTrain = rnn_Model.batchSizeTrain
        self.hiddenSize = rnn_Model.hiddenSize
        self.maxEpoch = rnn_Model.maxEpoch

        self.trainResult = rnn_Model.trainAcc_record
        self.testResult = rnn_Model.testAcc_record



class RNN_save_model(RNN_save_model_v0):
    def __init__(self, rnn_Model):
        super(RNN_save_model, self).__init__(rnn_Model)
        self.cluster_K = rnn_Model.cluster_K
        self.alpha_regularizer = rnn_Model.alpha_regularizer



class RNN_save_model_v2(RNN_save_model_v0):
    def __init__(self, rnn_Model):
        super(RNN_save_model_v2, self).__init__(rnn_Model)
        self.testResult_for = rnn_Model.testAcc_for_record
        self.testResult_bac = rnn_Model.testAcc_bac_record



class RNN_save_model_v3(RNN_save_model):
    def __init__(self, rnn_Model):
        super(RNN_save_model_v3, self).__init__(rnn_Model)
        self.param_dict = rnn_Model.param_dict



class RNN_save_model_v5(RNN_save_model_v3):
    def __init__(self, rnn_Model):
        super(RNN_save_model_v5, self).__init__(rnn_Model)
        self.trainResult2 = rnn_Model.trainAcc2_record
        self.testResult2 = rnn_Model.testAcc2_record
        # self.cluster_train = rnn_Model.cluster_train



class RNN_save_model_v4(RNN_save_model_v0):
    def __init__(self, rnn_Model):
        super(RNN_save_model_v4, self).__init__(rnn_Model)
        self.param_dict = rnn_Model.param_dict





def get_acc(epoch, output, label):

    output_prod = np.prod(output, 1)
    prodAcc = np.mean(np.equal(np.argmax(output_prod, 1), np.argmax(label, 1)))

    voteResult = []
    for i in output:
        temp = np.bincount(np.argmax(i, 1))
        voteResult.append(np.argmax(temp))
    voteResult = np.asarray(voteResult)
    voteAcc = np.mean(np.equal(voteResult, np.argmax(label, 1)))

    if (epoch + 1) % 3 == 0:
        print(
            "--epoch: %d    prod acc: %.4f    vote acc: %.4f   delta acc:%.4f" % (
                epoch + 1, prodAcc, voteAcc, prodAcc - voteAcc))


class temp_RNN_save_model():
    def __init__(self, rnn_Model):
        self.trainResult = rnn_Model.trainAcc_record
        self.testResult = rnn_Model.testAcc_record

        self.param_dict = rnn_Model.param_dict


if __name__ == '__main__':
    loadCluster(2)
