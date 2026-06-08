from configs import SEED, MODEL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math


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
            h = torch.relu(
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

#"Prefrontal cortex" of the model
class CustomRNN(nn.Module): #Takes input seq + Hidden state -> new hidden states
    def __init__(self, input_size=3, hidden_size=64, num_actions=2):
        super(CustomRNN, self).__init__() #initialises nn.Module
        self.input_size = input_size
        self.hidden_size = hidden_size

        #Define weights and biases
        self.W_xh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_xh = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))

        #Initialise the weights before use
        self.reset_parameters()

    def reset_parameters(self):
#        Xavier for inputs, Orthogonal for recurrent connections to sustain delay memory
        nn.init.normal_(self.W_xh, std=0.01)
        nn.init.normal_(self.W_hh, std=0.01)
        nn.init.zeros_(self.b_xh)
        nn.init.zeros_(self.b_hh)

    def forward(self, x, h_0=None):
        # x expected shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        #initialise first hidden state h_0 with zeros if not provided
        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h_0

        hidden_states = []

        #Unroll the RNN across time
        for t in range(seq_len):
            x_t = x[:, t, :] #get current timestep input

            h_t = torch.tanh(
                x_t @ self.W_xh.T + self.b_xh + 
                h_t @ self.W_hh.T + self.b_hh
            )

            hidden_states.append(h_t.unsqueeze(1))
    
        #return stacked hidden states and final hidden state
        return torch.cat(hidden_states, dim=1), h_t


#"Motor cortex/BG" of the model
class ActorCriticRNN(nn.Module): #Map hidden state to immediate physical action (lick/no lick)

    def __init__(self, input_size=3, hidden_size=64, num_actions=2):
        super(ActorCriticRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.rnn = CustomRNN(input_size, hidden_size)
        

        # The Actor and Critic heads (final output layers)
        self.actor_head = nn.Linear(hidden_size, num_actions) #Outputs policy (lick/ no lick)
        
        self.critic_head = nn.Linear(hidden_size, 1) #Outputs value estimate
        #Force 50/50 initial exploration
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias) 
    def forward(self, x, h_0=None):
        #Pass the sequence through custom RNN (including hidden state updates)
        hidden_states, h_n = self.rnn(x, h_0)
        
        #Actor Head
        action_logits = self.actor_head(hidden_states)
        raw_action_probs = torch.softmax(action_logits, dim=-1)
        
        epsilon = 0.05 
        num_actions = 2
        action_probs = (1 - epsilon) * raw_action_probs + (epsilon / num_actions)
        #Critic Head
        values = self.critic_head(hidden_states).squeeze(-1)
        
        #Returning h_n so we can pass it to the next chunk of time
        return action_probs, values, h_n
    
class RLModelWrapper(nn.Module):
    def __init__(self, rl_model):
        super().__init__()
        self.rl_model = rl_model
        # Match hidden size attributes for external references
        self.hidden_size = rl_model.hidden_size

    def forward(self, inputs):
        # Drop the 4th tracking channel if input comes from the SL batch generator
        if inputs.shape[-1] == 4:
            inputs = inputs[..., :3]
            
        action_probs, _, _ = self.rl_model(inputs)
        # Map Action 1 probability (Lick) to 'ys' to mimic the analog lick-rate output
        ys = action_probs[:, :, 1:2] 
        # Extract the hidden states directly from the internal custom RNN
        hs, _ = self.rl_model.rnn(inputs)
        return ys, hs

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.rl_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def eval(self):
        self.rl_model.eval()

    def train(self):
        self.rl_model.train()