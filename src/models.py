import torch
import torch.nn as nn
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

        self.bh = np.zeros((hidden_size, 1)) #hidden bias
        self.by = np.zeros((output_size, 1)) #output bias

    def forward(self, inputs):

        h = np.zeros((self.hidden_size, 1)) #creates an empty hidden state

        hs = [] #stores hidden states across time
        ys = [] #stores outputs across time

        for x in inputs:
            x = x.reshape(-1,1) #converts into a column vector [0,1,0,0] -> [[0],[1],[0],[0]]

            #Hidden state calculation (tanh nonlinearity)
            h = np.tanh(
                self.Wxh @ x + 
                self.Whh @ h +
                self.bh 
            )

            y = self.Why @ h + self.by #output at this timestep

            hs.append(h)
            ys.append(y)
        return ys, hs

