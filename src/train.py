from models import *
from task import *
import numpy as np
from configs import TRAINING, TASK
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions import Categorical


def train_model(rnn, dataset, lr, epochs, batches_per_epoch, batch_size, SIGMA=1.2, probe=False):

    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")    
    rnn.to(device)

    optimizer = optim.Adam(rnn.parameters(), lr=lr, weight_decay=1e-4)
    reversal_epoch = TRAINING["reversal_epoch"]
    criterion = nn.MSELoss(reduction='none') #use mean squared error
    
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)#0.99)

    # Unpack the probe data (for batch resolved plot)
    probe_inputs, _, probe_lengths, _, cues_probe = dataset["probe"]
    probe_inputs = probe_inputs.to(device)

    live_performance = {'A': [], 'B': [], 'C': []}

    loss_history = []
    rnn.train() # Set model to training mode

    idx_A = torch.tensor([i for i, c in enumerate(cues_probe) if c == 'A'])
    idx_B = torch.tensor([i for i, c in enumerate(cues_probe) if c == 'B'])
    idx_C = torch.tensor([i for i, c in enumerate(cues_probe) if c == 'C'])
    
    batch_idx = torch.arange(len(cues_probe))
    t_anticipation = (probe_lengths - 2) 


    for e in range(epochs):
        epoch_loss = 0
        
        if e == reversal_epoch:
            print("REVERSING STIMULUS VALUES - Resetting LR")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            #Restart the scheduler so LR decays from the top again
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        epoch_data = dataset["train"][e] #List of batch data for this epoch

        for b in range(batches_per_epoch):
            # Get the batch data
            padded_inputs, padded_targets, lengths, mask,cues = epoch_data[b]
            
            padded_inputs = padded_inputs.to(device)
            padded_targets = padded_targets.to(device)
            mask = mask.to(device)    

            optimizer.zero_grad() # Clear previous gradients
            
            # ys shape: (batch_size, seq_len, output_dim)
            ys, hs = rnn(padded_inputs) #runs rnn.forward 
    
            sq_error = criterion(ys, padded_targets) # Calculate element-wise MSE
            mask_expanded = mask.unsqueeze(-1) # Expand mask to match output dimensions (batch, seq, output_dim)
            masked_loss = (sq_error * mask_expanded).sum() / mask_expanded.sum() # Apply mask and calculate mean with only real (not padded) data
            
            masked_loss.backward() #runs all of the calculations for determining gradients
            
            nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0) #clips gradient to prevent explosion
            
            optimizer.step() #update the weights and biases
            
            epoch_loss += masked_loss.item()

            #PROBE THE NETWORK (Every batch)
            if probe:
                rnn.eval()
                with torch.no_grad():
                    ys_probe, _ = rnn(probe_inputs)
            
                    # Extract the specific time points for the whole batch at once
                    # Shape: (60)
                    vals = ys_probe[batch_idx, t_anticipation, 0]
                    
                    # Calculate means natively on the GPU, then pull just the final number (.item()) to CPU
                    live_performance['A'].append(vals[idx_A].mean().item())
                    live_performance['B'].append(vals[idx_B].mean().item())
                    live_performance['C'].append(vals[idx_C].mean().item())
                    
            rnn.train()

        avg_loss = epoch_loss / batches_per_epoch
        loss_history.append(avg_loss)
        print(f"Epoch: {e} | Loss: {avg_loss:.4f}")

        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        # Save checkpoints 
        if e == (reversal_epoch - 1): # Baseline once pretty much fully trained on normal stimuli
            torch.save(rnn.state_dict(), os.path.join(save_dir,"weights_baseline.pth"))
            
        elif e == (reversal_epoch + 5): # 5 epochs into the Reversal (Confusion)
            torch.save(rnn.state_dict(), os.path.join(save_dir,"weights_early_reversal.pth"))
            
        elif e == (epochs - 1): # The very end (Fully Reversed)
            torch.save(rnn.state_dict(), os.path.join(save_dir,"weights_final_reversal.pth"))
        if e % 2 == 0: #for animating a learning PCA
            torch.save(rnn.state_dict(), os.path.join(save_dir,f"weights_epoch_{e}.pth"))
        
        scheduler.step()

    save_path = os.path.join("checkpoints", "live_performance.pt")
    torch.save(live_performance, save_path)
    print(f"Saved live performance data to {save_path}")   

    return rnn, loss_history, live_performance

def compute_returns(rewards, gamma=0.99): 
    #calculates the discounted cumulative returns for the episode
    #Gamma=discount factor (prioritise near-term outcomes)
    #How much total future reward followed from this state/action
    returns = torch.zeros_like(rewards)
    running_return = torch.zeros(rewards.size(0), device=rewards.device)
    
    # Iterate backwards through time to calculate discounted returns
    for t in reversed(range(rewards.size(1))):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        
    return returns    

def train_rl_model(num_epochs=10000, batch_size=32, seq_len=15, lr=1e-3, gamma=0.95):
    # Initialize environment and model
    env = RLTask(batch_size=batch_size, seq_len=seq_len)
    model = ActorCriticRNN(input_size=3, hidden_size=64, num_actions=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)    

    loss_history = []
    live_performance = {'A': [], 'B': [], 'C': []}
    reversal_epoch = TRAINING.get("reversal_epoch", num_epochs // 2) #Defaults to halfway if none provided

    for epoch in range(num_epochs):
        #Get trial data
        inputs, stimuli = env.get_batch()
        action_probs, values = model(inputs) #Forward pass through the model

        dist = Categorical(action_probs)
        actions = dist.sample() # Shape = (batch_size, seq_len)
        log_probs = dist.log_prob(actions)

        rewards = env.evaluate_sequence(stimuli, actions)
        returns = compute_returns(rewards, gamma) #Calculates discounted returns

        advantage = returns - values.detach() #Outcome better or worse than expected?
        
        actor_loss = -(log_probs * advantage).mean() #Push probabilities up for good actions
        critic_loss = F.mse_loss(values, returns)

        # Entropy Bonus: Encourages exploration by penalizing absolute certainty
        entropy = dist.entropy().mean()
        entropy_coef = 0.01
        
        total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
        #Backprop
        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #Gradient clipping to prevent explosion
        optimizer.step()

        loss_history.append(total_loss.item())
   
        # Identify trial conditions by checking active channels across sequence length
        cue_A_mask = inputs[:, :, 0].sum(dim=1) > 0
        cue_B_mask = inputs[:, :, 1].sum(dim=1) > 0
        cue_C_mask = inputs[:, :, 2].sum(dim=1) > 0

        t_anticip = seq_len - 2  # Anticipation window step
        lick_prob = action_probs[:, t_anticip, 1] # Probability of choosing Lick (Action 1)

        live_performance['A'].append(lick_prob[cue_A_mask].mean().item() if cue_A_mask.any() else 0.0)
        live_performance['B'].append(lick_prob[cue_B_mask].mean().item() if cue_B_mask.any() else 0.0)
        live_performance['C'].append(lick_prob[cue_C_mask].mean().item() if cue_C_mask.any() else 0.0)
        
        # --- SAVING WEIGHTS AT CHECKPOINTS ---
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        
        if epoch == (reversal_epoch - 1):
            torch.save(model.state_dict(), os.path.join(save_dir, "weights_baseline.pth"))
        elif epoch == (reversal_epoch + 5):
            torch.save(model.state_dict(), os.path.join(save_dir, "weights_early_reversal.pth"))
        elif epoch == (num_epochs - 1):
            torch.save(model.state_dict(), os.path.join(save_dir, "weights_final_reversal.pth"))
            
        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"weights_epoch_{epoch}.pth"))

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:04d} | Total Loss: {total_loss.item():.4f} | Avg Lick Prob A: {live_performance['A'][-1]:.2f}")

    torch.save(live_performance, os.path.join("checkpoints", "live_performance.pt"))
    return model, loss_history, live_performance

def train_model_numpy(rnn, lr, epochs, batches_per_epoch, batch_size): 
    REVERSED = TASK["reversed"]
    #remember to pass ScratchRNNnumpy as rnn
    loss_history = []
    for e in range (epochs):
        epoch_loss = 0
        for b in range (batches_per_epoch): #eachx batch will update the parameters once
            padded_inputs, padded_targets, lengths, mask = generate_batch(batch_size=batch_size)
            #for now, convert tensors -> numpy arrays
            padded_inputs = padded_inputs.numpy()
            padded_targets = padded_targets.numpy()
            lengths = lengths.numpy()
            T = lengths.max() #longest trial
        
            mask = mask.numpy()

            ys, hs = rnn.forward(padded_inputs) #runs the RNN on this batch

            sq_error = (ys-padded_targets) ** 2 #computes the loss using mean squared error (MSE) - simple
            mask_expanded = mask[:, :, None] #adds a third dimension to mask, so it is the same shape
            masked_sq_error = sq_error * mask_expanded #removes error calculated on padding

            loss = np.sum(masked_sq_error) / np.sum(mask_expanded * ys.shape[2]) #computes the loss for this batch (divides by n)

            #gradients
            #change in weights
            dWxh = np.zeros_like(rnn.Wxh)
            dWhh = np.zeros_like(rnn.Whh)
            dWhy = np.zeros_like(rnn.Why)

            #change in biases
            dbh = np.zeros_like(rnn.bh)
            dby = np.zeros_like(rnn.by)
            
            #hidden gradient from future timestep
            dh_next = np.zeros((batch_size, rnn.hidden_size)) #shape (batch_size, hidden_size)

            for t in reversed(range(T)): #from last to first timestep
                dy = ys[:, t, :] - padded_targets[:, t, :] #gradient of loss with respect to outputs
                dy*= mask[:, t][:, None] #apply mask


                dWhy += dy.T @ hs[:, t, :] #gradient of loss with respect to output weights
                dby += np.sum(dy, axis=0, keepdims=True) #bias gradient

                dh = dy @ rnn.Why  + dh_next #total gradient flowing into current h from both output error + future timesteps
                dh_raw = (1 - hs[:, t, :]**2) * dh #(batch_size, hidden_size)

                dWxh += dh_raw.T @ padded_inputs[:, t, :]
                dbh += np.sum(dh_raw, axis=0, keepdims=True)

                if t> 0:
                    dWhh += dh_raw.T @ hs[:, t-1, :]

                dh_next = dh_raw @ rnn.Whh #gradient for next iteration (t-1)
            
            
            #first clip the weights to prevent explosion 
            for parameter in [dWxh, dWhh, dWhy, dbh, dby]:
                np.clip(parameter, -5, 5, out=parameter)
            #update the weights 
            rnn.Wxh -= lr * dWxh / batch_size #multiplies corresponding gradient by learning rate, scales to batch size
            rnn.Whh -= lr * dWhh / batch_size
            rnn.Why -= lr * dWhy / batch_size

            rnn.bh -= lr * dbh / batch_size
            rnn.by -= lr * dby / batch_size

            epoch_loss += loss
        avg_loss = epoch_loss/batches_per_epoch
        loss_history.append(avg_loss)
        print(f"Epoch: {e} | Loss: {avg_loss:.4f}")
    
    return rnn, loss_history #returns a fully trained RNN

