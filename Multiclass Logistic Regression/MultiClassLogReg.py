import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
from sklearn import datasets
from sklearn.decomposition import PCA
import time

x = None
x_test = None
y = None
W = 0  # initialize W 0
b = 0  # initialize bias 0
np.seterr(all='ignore')


def softmax(x):
    global y, W, b
    e = np.exp(x - np.max(x))  # prevent overflow

    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return np.divide(e , np.array([np.sum(e, axis=1)]).T)


def train_descent(lr=0.001, L2_reg=0.0):
    global x, y, W, b
    prob_y_given_x = softmax(np.dot(x, W) + b)
    d_y = y - prob_y_given_x

    # Modify weights. added regularization
    W += lr * np.dot(x.T, d_y) -  lr * L2_reg * np.square(W)
    b += lr * np.mean(d_y, axis=0)

# The cost function
def negative_log_likelihood():

    global x, y, W, b
    sigmoid_activation = softmax(np.dot(x, W) + b)

    cross_entropy = - np.mean(np.sum(y * np.log(sigmoid_activation) + (1 - y) * np.log(1 - sigmoid_activation), axis=1))
    return cross_entropy


def train_descent_optimization_func(_b_W):
    global x, y, W, b
    _b_W = np.reshape(_b_W, [len(W) + 1, len(W[0])])
    _b = _b_W[-1]
    _W = _b_W[:-1]

    prob_y_given_x = softmax(np.dot(x, _W) + _b)
    d_y = y - prob_y_given_x

    _Wgrad = np.dot(x.T, d_y)
    _bgrad = np.mean(d_y, axis=0)
    _return = np.vstack((_Wgrad, _bgrad))
    return np.reshape(_return, _return.size)


def neg_log_lh_optimization_func(_b_W):
    global b,W
    _b_W = np.reshape(_b_W,[len(W)+1, len(W[0])])
    b = _b_W[-1]
    W = _b_W[:-1]
    return negative_log_likelihood()

def scipy_fmin_bfgs():
    _b_W = np.vstack((W,b))
    _b_W = np.reshape(_b_W, np.size(_b_W))
    return optimize.fmin_bfgs(neg_log_lh_optimization_func, x0=_b_W, epsilon=0.000001, maxiter=10)

def scipy_check_grad():
    global b, W
    _b_W = np.vstack((W, b))
    _b_W = np.reshape(_b_W, _b_W.size)
    print _b_W
    optimize.check_grad(func=neg_log_lh_optimization_func, grad=train_descent_optimization_func, x0=_b_W)

def predict(x):
    global y, W, b
    return softmax(np.dot(x, W) + b)
def getLabelVector(n_classes=2, cl=2):
    tmp = np.zeros([n_classes])
    tmp[cl] = 1;
    return tmp
def loadDataset(type="default", pca_n_components=100):
    global  x, x_test, y, W, b
    if type=="hand_digits":
        print "Loading handwritten digits dataset..."
        data = datasets.load_digits()
        x = np.reshape(data.images, [len(data.images), 8*8])
        y = 0
        for label in data.target:
            if np.isscalar(y):
                y = getLabelVector(10, label)
            else:
                y = np.vstack((y, getLabelVector(10, label)))

    if type == "iris":
        print "Loading iris dataset..."
        iris = datasets.load_iris()
        x = iris.data
        y = 0
        x = x/10
        for label in iris.target:
            l = getLabelVector(3, label)
            if np.isscalar(y):
                y = l
            else:
                y = np.vstack((y, l))
    if type== "cifar10":
        data = cifar10_unpickle()
        x = data['data']
        y = 0
        for label in data['labels']:
            l = getLabelVector(10, label)
            if np.isscalar(y):
                y = l
            else:
                y = np.vstack((y, l))
        pca = PCA(n_components=pca_n_components)
        x = pca.fit_transform(x)

        x = x/np.amax(np.abs(x))

    #W = np.random.rand(len(x[0]), len(y[0]))
    #b = np.random.rand(len(y[0]))
    W = np.zeros([len(x[0]), len(y[0])],dtype="float")
    b = np.zeros([len(y[0])], dtype="float")

# Following method loads one-fifth of the whole dataset
# Data and the method were obtained from: https://www.cs.toronto.edu/~kriz/cifar.html
def cifar10_unpickle(file="cifar-10-batches-py\data_batch_1"):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def test_steepest_descent(learning_rate=0.00001, n_epochs=1000):
    global x, y, W, b

    cnt = 0
    last_loss = 0
    last_W = W
    last_b = b
    all_costs=[]
    all_accuracies=[]
    for epoch in xrange(n_epochs):
        train_descent(lr=learning_rate, L2_reg=0.0)
        cost = negative_log_likelihood()
        if cost!=cost:
            #if cost is nan
            W = last_W  # undo last step
            b = last_b
            learning_rate *= 0.5
            print "cost is nan... "
            continue

        if cnt > 5:
            break
        if cost == last_loss:
            cnt +=1
        # calculating and saving cost and accuracy for plotting purposes
        result = np.argmax(predict(x), axis=1) == np.argmax(y, axis=1)
        all_accuracies.append( float(np.sum(result))/len(result))
        all_costs.append(cost)

        if epoch%100==0:
            print "Training epoch {0}, cost is {1} , learning rate is: {2}".format(epoch, cost, learning_rate)
        if cost>last_loss:
            W = last_W # undo last step
            b = last_b
            learning_rate *= 0.5
        else:
            learning_rate *=1.02
        last_loss = cost
        last_W = W
        last_b = b

    #plt.plot(all_costs, label="Total Cost", color="b")

    #plt.plot(all_accuracies, label="Overall Accuracy", color="r")
    #plt.legend(loc="upper right")
    #plt.show()

def cross_validation_train(k=6):
    global x, x_test, y, W, b
    data_n_labels = np.concatenate((x, y), axis=1)
    np.random.shuffle(data_n_labels)
    y = data_n_labels[:, -10:len(data_n_labels)]     #-3 for iris dataset
    x = data_n_labels[:, :-10]                         #-10 for handwritten digits
    all_set = x
    y_test_labels = y
    all_labels = y
    range_ = int(len(all_set)/k)
    sum_of_results = 0.0
    for it in range(k):
        x_test = all_set[it*range_:(it+1)*range_]
        y_test_labels = all_labels[it * range_:(it + 1) * range_]
        x = np.delete(all_set,range(it * range_,(it + 1) * range_), 0)
        y = np.delete(all_labels, range(it * range_, (it + 1) * range_), 0)

        W = np.random.rand(len(x[0]), len(y[0]))
        b = np.random.rand(len(y[0]))

        test_steepest_descent(n_epochs=1000)
        result = np.argmax(predict(x_test),axis=1)== np.argmax(y_test_labels,axis=1)
        sum_of_results += float(np.sum(result)) / float(len(result))
    print "Total accuarcy was: ", sum_of_results/k
    x = all_set
    y = all_labels
    return sum_of_results/k



if __name__ == "__main__":
    loadDataset(type="hand_digits")
    #loadDataset(type="iris")
    #loadDataset(type="cifar10",pca_n_components=100)
    test_steepest_descent(n_epochs=1000, learning_rate=0.0001)

    #cross_validation_train(2)
    #print scipy_fmin_bfgs()
    #print scipy_check_grad()

    result = np.argmax(predict(x),axis=1)==np.argmax(y,axis=1)
    print float(np.sum(result))/len(result)
