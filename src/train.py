from models import ScratchRNN
from task import *
import numpy as np
from configs import TRAINING

lr = TRAINING["lr"]
 #to start with, just load one trial

rnn = ScratchRNN()
for z in range(100):
    for x in range(100):
        inputs, targets = generate_trial()
        ys, hs, xs = rnn.forward(inputs)

        loss = 0
        for i in range (len(ys)):
            loss += 0.5 * ((ys[i][0,0] - targets[i][0]) ** 2) #computes the loss using sum squared error (SSE) - simple

        

        #create empty derivative variables
        #change in weights
        dWxh = np.zeros_like(rnn.Wxh)
        dWhh = np.zeros_like(rnn.Whh)
        dWhy = np.zeros_like(rnn.Why)

        #change in biases
        dbh = np.zeros_like(rnn.bh)
        dby = np.zeros_like(rnn.by)

        # gradient flowing backward through time (change in loss over change in hidden state from the next timestep)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))): #from last to first timestep
            dy = ys[t] - targets[t] #gradient of loss with respect to outputs
            dWhy += dy @ hs[t].T #gradient of loss with respect to output weights
            dby += dy #bias gradient

            dh = rnn.Why.T @ dy + dh_next #total gradient flowing into current h from both output error + future timesteps
            dh_raw = (1 - hs[t]**2) * dh #derivative of tanh 

            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t-1].T

            dbh += dh_raw

            dh_next = rnn.Whh.T @ dh_raw #has shape (hidden_size, 1), gradient vector

        #gradient descent to compute new weights and update them
        rnn.Wxh -= lr * dWxh #multiplies corresponding gradient by learning rate, applies change
        rnn.Whh -= lr * dWhh
        rnn.Why -= lr * dWhy

        rnn.bh -= lr * dbh
        rnn.by -= lr * dby
    print("loss:", loss) #mainly for monitoring