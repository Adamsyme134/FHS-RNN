#import torch
#import torch.nn as nn
from configs import SEED, MODEL
import numpy as np


class ScratchRNN():

    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        output_size=1
        ):
        self.hidden_size = hidden_size
        np.random.seed(SEED)

        #Set up starting weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01 #input to hidden weights
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #recurrent hidden to hidden weights
        self.Why = np.random.randn(output_size, hidden_size) * 0.01 #hidden to output weights

        self.bh = np.zeros((1,hidden_size)) #hidden bias
        self.by = np.zeros((1,output_size)) #output bias

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        max_length = inputs.shape[1]

        h = np.zeros((batch_size,self.hidden_size)) #creates an empty hidden state

        hs = [] #stores hidden states across time
        ys = [] #stores outputs across time
        
        for t in range (max_length): #loops through every timestep
            xt = inputs[: , t, :] #gets each input with shape (batch_size, input_size), so every input at time t from across all batches
            
           
            #Hidden state calculation (tanh nonlinearity)
            h = np.tanh(
                xt @ self.Wxh.T + 
                h @ self.Whh.T +
                self.bh 
            )

            yt = h @ self.Why.T + self.by #output at this timestep

            hs.append(h.copy())
            ys.append(yt.copy())

        ys = np.stack(ys, axis =1) #gives outputs in shape (batch_size, max_length, output_size)
        hs = np.stack(hs, axis =1)
        return ys, hs

