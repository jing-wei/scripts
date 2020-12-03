
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot",
                 normalization=False
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
            if normalization:
                self.normalize()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n])) 
            n_in = all_dims[layer_n - 1] 
            n_out = all_dims[layer_n] 
            self.weights[f"W{layer_n}"] = np.random.uniform(-np.sqrt(6/(n_in + n_out)), np.sqrt(6/(n_in + n_out)), (n_in, n_out)) 
            # dim to match grader

    def relu(self, x, grad=False):
        if grad:
            return (x>0).astype('int32')
        else: 
            return x*(x>0).astype('int32')
        
        

    def sigmoid(self, x, grad=False):
        if grad:
            return np.exp(-x)/(1+np.exp(-x))**2 
        else: 
            return 1/(1+np.exp(-x)) 

    def tanh(self, x, grad=False):
        if grad:
            return 1 - ((np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)))**2 
        else: 
            return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) 
        

    def leakyrelu(self, x, grad=False):
        alpha = 0.01
        tmp = np.ones(x.shape) 
        tmp[x <= 0] = alpha
        if grad:
            return tmp 
        else: 
            return tmp*x

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        elif self.activation_str == "leakyrelu":
            return self.leakyrelu(x, grad) 
        else:
            raise Exception("invalid")
        

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        if len(x.shape)==1:
            myX = np.expand_dims(x,axis=0)
        else:
            myX = x
        C = np.max(myX)
        myExps = np.exp(myX-C)
        mySum = np.expand_dims(np.sum(myExps,axis=1),axis=1)
        mySoft = myExps/np.matmul(mySum,np.ones((1,myX.shape[1])))
        if mySoft.shape[0] == 1:
            mySoft = np.squeeze(mySoft)
        return mySoft
            

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i 
        for layer_n in range(1, self.n_hidden + 1):
            cache[f"A{layer_n}"] = np.matmul(cache[f"Z{layer_n - 1}"], self.weights[f"W{layer_n}"]) +  self.weights[f"b{layer_n}"]
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"], grad=False) 
            
        cache[f"A{self.n_hidden + 1}"] = np.matmul(cache[f"Z{self.n_hidden}"], self.weights[f"W{self.n_hidden+1}"]) +  self.weights[f"b{self.n_hidden+1}"]
        cache[f"Z{self.n_hidden + 1}"] = self.softmax(cache[f"A{self.n_hidden+1}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        batch_size = labels.shape[0] 
        grads[f"dA{self.n_hidden + 1}"] = output - labels 
        for layer_n in [l for l in range(1, self.n_hidden + 2)][::-1]: 
            grads[f"dW{layer_n}"] = np.matmul(cache[f"Z{layer_n-1}"].transpose(), grads[f"dA{layer_n}"])/batch_size
            grads[f"db{layer_n}"] = np.expand_dims(np.mean(grads[f"dA{layer_n}"],axis=0),axis=0)
            
            if layer_n>1:
                grads[f"dZ{layer_n-1}"] = np.matmul(grads[f"dA{layer_n}"], self.weights[f"W{layer_n}"].transpose())
                grads[f"dA{layer_n-1}"] = grads[f"dZ{layer_n-1}"] * self.activation(cache[f"A{layer_n-1}"],grad=True)
            
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - self.lr * grads[f"dW{layer}"] 
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - self.lr * grads[f"db{layer}"] 

    def one_hot(self, y):
        y_oh = np.zeros((y.shape[0], self.n_classes)) 
        for i in range(y.shape[0]): 
            y_oh[i, y[i]] = 1
        return y_oh

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        return -np.mean(np.sum(np.log(prediction)*labels, axis=1))

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache = self.forward(minibatchX) 
                grads = self.backward(cache, minibatchY) 
                self.update(grads) 

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test) 
        return test_loss, test_accuracy 
    

    def normalize(self):
        # different here
        # WRITE CODE HERE
        # compute mean and std along the first axis
        myMean = np.mean(self.train[0],axis=0)
        myStdv = np.std(self.train[0],axis=0)
        # Normalize training data
        self.train = ((self.train[0]-myMean)/myStdv , self.train[1]) 
        # Normalize test data
        self.test = ((self.test[0]-myMean)/myStdv , self.test[1])
        # Normalize validation data
        self.valid = ((self.valid[0]-myMean)/myStdv , self.valid[1])
        
    
##########################################################
# run on SVHN
model_norm = NN(hidden_dims=(512, 256), 
   datapath='../svhn.pkl', # adjust to path of input data .pkl
   n_classes=10,
   epsilon=1e-6,
   lr=0.03,
   batch_size=100,
   seed=0,
   activation="relu",
   normalization=True )


model_norm.train_loop(n_epochs = 30)

import matplotlib.pyplot as plt 
import seaborn as sns 
import time 
import os 

# 30 epochs
x = [i for i in range(30)] 

fig = plt.figure() 
ax = fig.add_subplot(111)
ax.plot(x, model_norm.train_logs['train_accuracy'], label='train accuracy') 
ax.plot(x, model_norm.train_logs['validation_accuracy'], label = 'validation accuracy') 
ax.legend()
ax.set(title = 'training and validation accuracies', 
      ylabel='accuracies', 
      xlabel = 'n_epochs')
plt.show()

# 30 epochs
x = [i for i in range(30)] 

fig = plt.figure() 
ax = fig.add_subplot(111)
ax.plot(x, model_norm.train_logs['train_loss'], label='train loss') 
ax.plot(x, model_norm.train_logs['validation_loss'], label = 'validation loss') 
ax.legend()
ax.set(title = 'training and validation losses', 
      ylabel='losses', 
      xlabel = 'n_epochs')
plt.show()


myNN_deep = NN(hidden_dims=(512, 120, 120, 120, 120, 120, 120),
datapath='../svhn.pkl',
n_classes=10,
lr=0.03,
batch_size=100,
seed=0,
activation="relu")

myNN_deep.train_loop(n_epochs = 30) 

# 30 epochs
x = [i for i in range(30)] 

fig = plt.figure() 
ax = fig.add_subplot(111)
ax.plot(x, myNN_deep.train_logs['train_accuracy'], label='train accuracy') 
ax.plot(x, myNN_deep.train_logs['validation_accuracy'], label = 'validation accuracy') 
ax.legend()
ax.set(title = 'training and validation accuracies', 
      ylabel='accuracies', 
      xlabel = 'n_epochs')
plt.show()

# 30 epochs
x = [i for i in range(30)] 

fig = plt.figure() 
ax = fig.add_subplot(111)
ax.plot(x, myNN_deep.train_logs['train_loss'], label='train loss') 
ax.plot(x, myNN_deep.train_logs['validation_loss'], label = 'validation loss') 
ax.legend()
ax.set(title = 'training and validation losses', 
      ylabel='losses', 
      xlabel = 'n_epochs')
plt.show()