# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .layers import Dense
from .loss_func import LossFunction
from .utils import Activation
from .model import Model, Sequence
from .progressbar import ProgressBar

class GRU(object):
    
    def __init__(self,
                 hidden_size,
                 lr=0.01,
                 optimizer='momentem'):
        
        # learning rate
        self.lr = lr
        # parameter mu
        self.mu = None
        # save mu to back propagation
        self.mu_cache = []
        # parameter r
        self.r = None
        # save r to back propagation
        self.r_cache = []
        # parameter h hat
        self.h_hat = None
        # save h_hat to back propagation
        self.h_hat_cache = []
        # optimizer name
        self.optimizer = optimizer
        # hidden layer number
        self.hidden_size = hidden_size
        # h of the previous time 
        self.h_prev = None
        # save previous h to back propagation
        self.h_prev_cache = []
        
    def init_gate_parameters(self, A_prev):
        """
            init the gate's parameter, like: mu \ r \ h_hat \ y
            I define these parameters as a Dense layer with activation function
        """
        if self.mu is None and self.r is None:
            n, m= A_prev.shape
            self.h_prev = np.zeros((self.hidden_size, m))
            self.mu = Sequence(training_data=None,
                               learning_rate=self.lr,
                               optimizer=self.optimizer,
                               layers=[Dense(self.hidden_size),
                                       Activation('sigmoid')])
            self.r = Sequence(training_data=None,
                              learning_rate=self.lr,
                              optimizer=self.optimizer,
                              layers=[Dense(self.hidden_size),
                                      Activation('sigmoid')])
            self.h_hat = Sequence(training_data=None,
                                  learning_rate=self.lr,
                                  optimizer=self.optimizer,
                                  layers=[Dense(self.hidden_size),
                                          Activation('tanh')])
        
    def one_step_forward(self, A_prev):
        """
            one step of the forward propagation
        """
        # init the previous loss of h
        self.delta_h_prev = 0.
        # init the gate's parameters
        self.init_gate_parameters(A_prev)
        # save h of the previous time
        self.h_prev_cache.append(self.h_prev)
        # do the operation to get mu
        mu_val = self.mu.forward(np.vstack((self.h_prev, A_prev)))
        # save the current mu
        self.mu_cache.append(mu_val)
        # do the operation to get r
        r_val = self.r.forward(np.vstack((self.h_prev, A_prev)))
        # save the current r
        self.r_cache.append(r_val)
        # do the operation to get h_hat
        h_hat_val = self.h_hat.forward(np.vstack((self.h_prev * r_val, A_prev)))
        # save the current h_hat
        self.h_hat_cache.append(h_hat_val)
        # get the current h
        h_val = (1. - mu_val) * self.h_prev + mu_val * h_hat_val
        # save the current h as previous h to use at the next time
        self.h_prev = h_val

        return h_val
    
    def forward(self, A_prev):
        """
            the whole forward propagation
        """
        assert len(A_prev.shape) == 3, "input data shape is wrong, expect 3dim"
        # get the input data's shape
        n, m, Tx = A_prev.shape
        # save the time series length
        self.time_series_length = Tx
        # save the hidden state
        h_vals = []
        # begin loop by time
        for i in range(Tx):
            h_val = self.one_step_forward(A_prev[:,:,i])
            h_vals.append(h_val)
        # return the hidden states
        return h_vals
        
    
    def one_step_backward(self, delta, t=0):
        """
            one step of the back propagation
        """
        # get mu at the nearest time
        mu_val = self.mu_cache.pop()
        # get r at the nearest time
        r_val = self.r_cache.pop()
        # get h_hat at the nearest time
        h_hat_val = self.h_hat_cache.pop()
        # get h_prev at the nearest time
        h_prev = self.h_prev_cache.pop()
        #--------------------begin back propagation--------------------
        # get the error of h by output layer's backpropagation
        delta_h = self.delta_h_prev + delta
        # get the error of mu
        delta_mu = delta_h * (h_hat_val - h_prev)
        # get the error of h_hat
        delta_h_hat = delta_h * mu_val
        # get the error of previous h
        self.delta_h_prev = delta_h * (1. - mu_val)
        # get the error of r with x by h_hat's layer
        delta_r_h = self.h_hat.backward(delta_h_hat, t)
        # slime the first hidden_num of delta_r_h to get the error of r
        delta_r = delta_r_h[:self.hidden_size, :] * h_prev
        delta_x = delta_r_h[self.hidden_size:, :]
        # get the error of previous h by h_hat
        self.delta_h_prev += delta_r_h[:self.hidden_size, :] * r_val
        # get the error of previous h by r
        delta_r_hx = self.r.backward(delta_r, t)
        self.delta_h_prev += delta_r_hx[:self.hidden_size, :]
        delta_x += delta_r_hx[self.hidden_size:, :]
        # get the error of previous h by mu
        delta_mu_hx = self.mu.backward(delta_mu, t)
        self.delta_h_prev += delta_mu_hx[:self.hidden_size, :]
        delta_x += delta_mu_hx[self.hidden_size:, :]
        # return previous h
        return delta_x
    
    def backward(self, dZ, t=0):
        """
            the whole back propagation
        """
        assert len(dZ.shape) != 1, "the error of the post layer is wrong"
        # get the sample of the errors
        m = dZ.shape[1]
        # save the input error
        dZs_prev = []
        # time series length
        times = self.time_series_length
        if len(dZ.shape) == 2:
            # the error shape like : n, m
            # begin loop to back propagate the error
            for i in range(times):
                delta = dZ if i == 0 else 0.
                # input the output error is zero
                dZ_prev = self.one_step_backward(delta, t)
                dZs_prev.append(dZ_prev)
                
        elif len(dZ.shape) == 3:
            # the error shape like : n, m, Tx
            # begin loop to back propagate the error
            for i in range(times):
                # input the output error
                 dZ_prev = self.one_step_backward(dZ[:,:,times-1-i], t)
                 dZs_prev.append(dZ_prev)
                 
        return np.array(dZs_prev).reshape((-1, m, times))
    
    def update_learning_rate(self, lr):
        """
            update the learning rate
        """
        self.lr = lr
        self.mu.learning_rate = lr
        self.r.learning_rate = lr
        self.h_hat.learning_rate = lr

class GRUI(GRU):
    
    def __init__(self,
                 hidden_size,
                 lr=0.01,
                 time_lag=0.,
                 optimizer='momentem'):
        # time lag parameter
        self.time_lag = time_lag
        # the important parameter in the GRUI
        self.beta = None
        # save the beta
        self.beta_cache = []
        # call the GRU's init func
        super().__init__(hidden_size, lr, optimizer)
        
    def init_gate_parameters(self, A_prev):
        # call the GRU's init func
        super().init_gate_parameters(A_prev)
        # init the beta
        if self.beta is None:
            n, m = A_prev.shape
            self.beta = Sequence(training_data=None,
                                 learning_rate=self.lr,
                                 optimizer=self.optimizer,
                                 layers=[Dense(self.hidden_size),
                                         Activation('relu')])
    
    def one_step_forward(self, A_prev):
        """
            P.S. add beta to the one step forward propagation 
        """
        # init the parameters
        self.init_gate_parameters(A_prev)
        # get the intput data's shape
        n, m = A_prev.shape
        # generate the time lag matrix
        time_lag_matrix = np.ones((n, m)) * self.time_lag
        # get the beta
        beta_val = 1. / np.exp(self.beta.forward(time_lag_matrix))
        # save the beta
        self.beta_cache.append(beta_val)
        # update the h
        h_prev = beta_val * self.h_prev
        # call the GRU's forward propagation
        return super().one_step_forward(h_prev)
    
    def ont_step_backward(self, delta, t=0):
        """
            P.S. add beta to the one step back propagation
        """
        # call the GRU's back propagation
        h_prev = super().one_step_backward(delta)
        # get beta at the nearest time
        beta_val = self.beta_cache.pop()
        # get the error of beta
        delta_beta_val = self.delta_h_prev * h_prev
        # update the beta's layer parameters
        self.beta.backward(-beta_val * delta_beta_val, t)
        # get the error of previous h
        self.delta_h_prev = self.delta_h_prev * beta_val
        # return the previous h
        return h_prev
        

class RNN(Model):

    def __init__(self,
                 output_layers,
                 lr=0.01,
                 mode='m2m',
                 decay_rate=0.,
                 hidden_size=16,
                 cell_type='GRU',
                 optimizer='momentem',
                 loss_func='mean_square_error'):
        # the mode of handle the squence
        # m2m : many input to many output
        # m2o : many input to one output
        # o2m : one input to many output
        self.mode = mode
        # learning rate
        self.learning_rate = lr
        # hidden size
        self.hidden_size = hidden_size
        # the decay rate of the learning rate
        self.decay_rate = decay_rate
        # the loss function
        self.lossFunction = LossFunction(loss_func)
        # output layer
        self.output_layer = Sequence(training_data=None,
                                     learning_rate=lr,
                                     optimizer=optimizer,
                                     layers=output_layers)
        # output cache
        self.output_cache = []
        # the RNN's cell
        if cell_type == 'GRU':
            self.cell = GRU(hidden_size,
                            lr=lr,
                            optimizer=optimizer)
        elif cell_type == 'GRUI':
            self.cell = GRUI(hidden_size,
                             lr=lr,
                             optimizer=optimizer)
    
    def m2m_forward(self, A_prev):
        # get the input shape
        _, m, Tx = A_prev.shape
        # GRU cell forward propagation
        h_vals = self.cell.forward(A_prev)
        # output values
        output = []
        # input hidden state to output layer
        for j in range(Tx):
            index = Tx - j - 1
            py = self.output_layer.forward(h_vals[index])
            output.insert(0, py)
        return output
    
    def m2m_backward(self, dZ, t=0):
        # get the error shape
        _, m, Tx = dZ.shape
        # init the delta h
        delta_h = np.zeros((self.hidden_size, m, Tx))
        # output layer back propagation
        for j in range(Tx):
            delta_h[:,:,j] = \
                    self.output_layer.backward(dZ[:,:,j], t)
        # RNN cell
        return self.cell.backward(delta_h, t)
    
    def m2o_forward(self, A_prev):
        # get the input shape
        _, m, Tx = A_prev.shape
        # GRU cell forward propagation
        h_vals = self.cell.forward(A_prev)
        # output values
        output = []
        # input hidden state to output layer
        py = self.output_layer.forward(h_vals[-1])
        output.append(py)
        return output
    
    def m2o_backward(self, dZ, t=0):
        # get the error shape
        _, m, _ = dZ.shape
        # output layer back propagation
        delta_h = self.output_layer.backward(dZ.reshape((-1, m)), t)
        # RNN cell
        return self.cell.backward(delta_h, t)
    
    def o2m_forward(self, A_prev):
        # get the input shape
        n, m, Tx = A_prev.shape
        # get the first time data
        x = A_prev[:,:,0]
        # output values
        output = []
        for i in range(1,Tx):
            # RNN cell forward propagation
            h_val = self.cell.forward(x.reshape((n,m,1)))[0]
            py = self.output_layer.forward(h_val)
            output.append(py)
            x = py
        return output
    
    def o2m_backward(self, dZ, t=0):
        # get the error shape
        _, m, Tx = dZ.shape
        # error of the post layer
        dZ_prev = None
        for i in range(Tx-1):
            index =Tx - i - 1
            # output layer back propagation
            delta_y = self.output_layer.backward(dZ[:,:,index], t)
            # RNN cell back propagation
            dZ_prev = self.cell.backward(delta_y.reshape((-1,m,1)), t)
        
        return dZ_prev[:,:,0]

    def forward(self, A_prev):
        
        return getattr(self, '{}_forward'.format(self.mode))(A_prev)
        
    def backward(self, dZ, t=0):
        
        return getattr(self, '{}_backward'.format(self.mode))(dZ, t)
    
    def train(self, X, y, epoch=1, mini_batch_size=100, verbose=False):
        # get the input data's shape
        # xn : the dimension of the data
        # xm : the sample's number
        xn, xm, Tx = X.shape
        # get the output data's shape
        yn, ym, Ty = (*y.shape,1) if len(y.shape) == 2 else y.shape
        # confirm the data's dimension is right
        assert xm == ym, "input data sample's number is error, x : {} y : {}".format(xm, ym)
        y = y.reshape((yn, ym, Ty))
            
        losses = []
        # begin epoch
        for cnt in range(1, epoch+1):
            # save the losses
            sum_loss = []
            acc = 0.
            # begin iteration
            # go through the whole data by mini_batch
            for i in range(xm // mini_batch_size):
                # tmp loss
                loss = 0.
                # generate the batch data
                x = X[:, i*mini_batch_size:(i+1)*mini_batch_size, :]\
                        .reshape((xn, mini_batch_size, -1))
                _y = y[:, i*mini_batch_size:(i+1)*mini_batch_size, :]\
                        .reshape((yn, mini_batch_size, -1))
                delta_y = np.zeros_like(_y)
                # begin forward propagation
                py = self.forward(x)
                acc += np.mean(py[0] * _y[:,:,0])
                # calc the output error
                for i in range(len(py)):
                    loss += self.lossFunction.getLoss(_y[:,:,i], py[i])
                    delta_y[:,:,i] = py[i] - _y[:,:,i]
                # save the current loss
                sum_loss.append(loss)
                # begin backward propagation
                self.backward(delta_y, cnt)
            loss = np.mean(sum_loss)
            # update the learning rate if need
            self.updateLearningrate(cnt)
            # collecting the mean loss value
            losses.append(loss)
            # if print the loss info
            if verbose:
#                if (cnt % (epoch / 50) == 0):
                print("epoch{0} loss:{1} acc:{2}".format(cnt, '%.6f'%loss, '%.6f'%(acc/(xm//mini_batch_size))))
        # plot the loss curve
        plt.plot(np.arange(epoch), losses)
        plt.title('loss')

    def updateLearningrate(self, t):
        
        self.learning_rate = 1. / (1. + self.decay_rate * t) * self.learning_rate
        self.cell.update_learning_rate(self.learning_rate)
        
        