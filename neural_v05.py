import numpy as np
import random
import sys

class NeuralNetwork:
    """A simple example class"""

    # init_weight = 0.15 # initial weights of neuron connections
    J_old = 1e15 # initial value of the function to be minimized

    def __init__(self, num_inputs, input_data, num_outputs, output_data, layers, layer_neurons, bias, activation_function):
        self.N = input_data.shape[0] # data length
        self.ne = num_inputs
        self.in_data = np.resize(input_data,(self.N,1))
        self.no = num_outputs
        self.out_data = output_data
        self.layers = layers
        self.nm = layer_neurons
        if bias:
            self.ne += 1
            self.in_data = np.column_stack((self.in_data,np.ones(self.N)))
        self.fun = activation_function

    def set_w1(self, init_value = 0.15):
        # self.v = np.random.normal(init_value, 0.6*init_value, (self.ne,self.nm))
        self.w1 = init_value*np.random.random((self.ne,self.nm))

    def get_w1(self):
        return self.w1

    def set_w2(self, init_value = 0.15):
        # self.w = np.random.normal(init_value, 0.6*init_value, (self.nm,self.no))
        self.w2 = init_value*np.random.random((self.nm,self.no))

    def get_w2(self):
        return self.w2

    def sigmoid(self, m,first_derivative=False):
        if self.fun == "sigmoid_1":
            return 1/(1 + np.exp(-m))
        elif self.fun == "sigmoid_2":
            return 2/(1 + np.exp(-m)) - 1
        elif self.fun == "gaussian":
            return np.exp(-np.power(m,2))
        else:
            sys.exit("Wrong activation function - check name")

    def tanh(self, n, first_derivative=True):
        if self.fun == "sigmoid_1":
            return n*(1-n)
        elif self.fun == "sigmoid_2":
            return (1-n*n)/2
        else:
            sys.exit("Wrong activation function - check name")

    def execute(self, rate = 0.1, max_iter = 10000):
        self.set_w1()
        self.set_w2()
        y=self.out_data
        X=self.in_data
        N=int(X.shape[0])
        reg_coeff=1e-6
        losses = []
        accuracies=[]
        # Initialize weights:
        np.random.seed(2017)
        w1 = 2.0*np.random.random((self.ne, self.nm))-1.0      #w0=(2,self.nm)
        w2 = 2.0*np.random.random((self.nm, self.no))-1.0     #w1=(self.nm,2)

        #Calibratring variances with 1/sqrt(fan_in)
        w1 /= np.sqrt(self.ne)
        w2 /= np.sqrt(self.nm)
        for i in range(max_iter):

            index = np.arange(X.shape[0])[:N]
            #is want to shuffle indices: np.random.shuffle(index)

            #---------------------------------------------------------------------------------------------------------------
            # Forward step:
            h1 = self.sigmoid(np.matmul(X[index], w1))                   #(N, 3)
            logits = self.sigmoid(np.matmul(h1, w2))                     #(N, 2)
            probs = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
            h2 = logits

            #---------------------------------------------------------------------------------------------------------------
            # Definition of Loss function: mean squared error plus Ridge regularization
            L = np.square(y[index]-h2).sum()/(2*N) + reg_coeff*(np.square(w1).sum()+np.square(w2).sum())/(2*N)

            losses.append([i,L])
            
            #---------------------------------------------------------------------------------------------------------------
            # Backward step: Error = W_l e_l+1 f'_l
            #       dL/dw2 = dL/dh2 * dh2/dz2 * dz2/dw2
            dL_dh2 = -(y[index] - h2)                               #(N, 2)
            dh2_dz2 = self.sigmoid(h2, first_derivative=True)            #(N, 2)
            dz2_dw2 = h1                                            #(N, self.nm)
            #Gradient for weight2:   (self.nm,N)x(N,2)*(N,2)
            dL_dw2 = dz2_dw2.T.dot(dL_dh2*dh2_dz2) + reg_coeff*np.square(w2).sum()

            #dL/dw1 = dL/dh1 * dh1/dz1 * dz1/dw1
            #       dL/dh1 = dL/dz2 * dz2/dh1
            #       dL/dz2 = dL/dh2 * dh2/dz2
            
            
            dL_dz2 = dL_dh2 * dh2_dz2                               #(N, 2)
            dz2_dh1 = w2                                            #z2 = h1*w2
            dL_dh1 =  dL_dz2.dot(dz2_dh1.T)                         #(N,2)x(2, self.nm)=(N, hidden_dim)
            dh1_dz1 = self.sigmoid(h1, first_derivative=True)            #(N,self.nm)
            dz1_dw1 = X[index]                                      #(N,2)
            #Gradient for weight1:  (2,N)x((N,self.nm)*(N,self.nm))
            dL_dw1 = dz1_dw1.T.dot(dL_dh1*dh1_dz1) + reg_coeff*np.square(w1).sum()

            #weight updates:
            w2 += -rate*dL_dw2
            w1 += -rate*dL_dw1
            if True: #(i+1)%1000==0:
                y_pred = inference(X, [w1, w2])
                y_actual = np.argmax(y, axis=1)
                accuracy = np.sum(np.equal(y_pred,y_actual))/len(y_actual)
                accuracies.append([i, accuracy])

            if (i+1)% 10000 == 0:
                print('Epoch %d\tLoss: %f Average L1 error: %f Accuracy: %f' %(i, L, np.mean(np.abs(dL_dh2)), accuracy))
                save_filepath = './scratch_mlp/plots/boundary/image_%d.png'%i
                text = 'Batch #: %d    Accuracy: %.2f    Loss value: %.2f'%(i, accuracy, L)
                utils.plot_decision_boundary(X, y_actual, lambda x: inference(x, [w1, w2]),
                                            save_filepath=save_filepath, text = text)
                save_filepath = './scratch_mlp/plots/loss/image_%d.png' % i
                utils.plot_function(losses, save_filepath=save_filepath, ylabel='Loss', title='Loss estimation')
                save_filepath = './scratch_mlp/plots/accuracy/image_%d.png' % i
                utils.plot_function(accuracies, save_filepath=save_filepath, ylabel='Accuracy', title='Accuracy estimation')
