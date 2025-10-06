import numpy as np

###### useful functions #############################################################################################################################
def positive(x, th, th0):
   return np.sign(np.dot(np.transpose(th), x) + th0)
    
def score(data, labels, ths, th0s):
   pos = np.sign(np.dot(np.transpose(ths), data) + np.transpose(th0s)) 
   return np.sum(pos == labels, axis = 1, keepdims = True)


###### actual perceptron ############################################################################################################################
def perceptron(data, labels, params = {}):
    # if T not in params, default to 100
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
    return theta, theta_0

# Regular perceptron can be somewhat sensitive to the most recent examples that it sees. Instead, averaged perceptron produces 
# a more stable output by outputting the average value of th and th0 across all iterations
def averaged_perceptron(data, labels, params = {}):
    # if T not in params, default to 100
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    thetas = np.zeros((d, 1)); theta_0s = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
            thetas = thetas + theta
            theta_0s = theta_0s + theta_0
    return thetas/(n*T), theta_0s/(n*T)

###### evaluating ###################################################################################################################################
def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    (d, n) = data_test.shape
    th, th0 = learner(data_train, labels_train)
    
    result = score(data_test, labels_test, th, th0)
    return result/n

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    total = 0
    for i in range(it):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test)
        total += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
    return total/it   

def xval_learning_alg(learner, data, labels, k):
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
    return score_sum/k