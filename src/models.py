from configs import SEED, MODEL
import numpy as np
import torch
import torch.nn as nn


class ScratchRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, output_size=1):
        super(ScratchRNN, self).__init__() 
        self.hidden_size = hidden_size
        
        # Set PyTorch seed for reproducibility of random numbers
        torch.manual_seed(SEED)

        # Define weights + biases randomly to start
        self.Wxh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Why = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01) #kaiming normal initialisation? CHANGE?

        self.bh = nn.Parameter(torch.zeros(1, hidden_size))
        self.by = nn.Parameter(torch.zeros(1, output_size))

    def forward(self, inputs):
        # inputs shape: (batch_size, max_length, input_size)
        batch_size = inputs.shape[0]
        max_length = inputs.shape[1]
        device = inputs.device # Ensure h is on the same device (CPU/GPU) as inputs

        #hidden states reset to 0 each batch
        h = torch.zeros(batch_size, self.hidden_size, device=device)

        hs = [] 
        ys = [] 
        
        for t in range(max_length):
            xt = inputs[:, t, :] # (batch_size, input_size)
            
         
            #calculate new hidden state, using tanh as before
            h = torch.tanh(
                xt @ self.Wxh.t() + 
                h @ self.Whh.t() +
                self.bh 
            )

            yt = h @ self.Why.t() + self.by #calculate new output at t

            #store the hidden states and outputs over time
            hs.append(h)
            ys.append(yt)

        #make sure ys and hs have the right shape
        ys = torch.stack(ys, dim=1) 
        hs = torch.stack(hs, dim=1)
        
        return ys, hs

class ScratchRNNnumpy():

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

