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




def train_model(rnn, dataset, lr, epochs, batches_per_epoch, batch_size, SIGMA=1.2, probe=False, save_dir="checkpoints"):

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

            optimizer = optim.Adam(rnn.parameters(), lr=lr, weight_decay=1e-4)
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
            sq_error = criterion(ys, padded_targets)

            importance_weights = torch.where(padded_targets > 0.1, 10.0, 1.0)
            sq_error = sq_error * importance_weights
            mask_expanded = mask.unsqueeze(-1) 
            masked_loss = (sq_error * mask_expanded).sum() / mask_expanded.sum()
            
            masked_loss.backward()
            optimizer.step()
             #runs all of the calculations for determining gradients
            
            nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0) #clips gradient to prevent explosion
            
            optimizer.step() #update the weights and biases
            
            epoch_loss += masked_loss.item()
            t_peak = probe_lengths - 1
            #PROBE THE NETWORK (Every batch)
            if probe:
                rnn.eval()
                with torch.no_grad():
                    ys_probe, _ = rnn(probe_inputs)
            
                    # Extract the specific time points for the whole batch at once
                    # Shape: (60)
                    vals = ys_probe[batch_idx, t_peak, 0]
                    
                    # Calculate means natively on the GPU, then pull just the final number (.item()) to CPU
                    live_performance['A'].append(vals[idx_A].mean().item())
                    live_performance['B'].append(vals[idx_B].mean().item())
                    live_performance['C'].append(vals[idx_C].mean().item())
                    
            rnn.train()

        avg_loss = epoch_loss / batches_per_epoch
        loss_history.append(avg_loss)
        print(f"Epoch: {e} | Loss: {avg_loss:.4f}")

        
        os.makedirs(save_dir, exist_ok=True)
        # Save checkpoints 
        if e == (reversal_epoch - 1) or (e == epochs - 1 and reversal_epoch > epochs): # Baseline once pretty much fully trained on normal stimuli
            torch.save(rnn.state_dict(), os.path.join(save_dir,"weights_baseline.pth"))
            
        elif e == (reversal_epoch + 5): # 5 epochs into the Reversal (Confusion)
            torch.save(rnn.state_dict(), os.path.join(save_dir,"weights_early_reversal.pth"))
            
        elif e == (epochs - 1): # The very end (Fully Reversed)
            torch.save(rnn.state_dict(), os.path.join(save_dir,"weights_final_reversal.pth"))
        if e % 2 == 0: #for animating a learning PCA
            torch.save(rnn.state_dict(), os.path.join(save_dir,f"weights_epoch_{e}.pth"))
        
        scheduler.step()

    save_path = os.path.join(save_dir, "live_performance.pt")
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

def compute_returns_continuous(rewards, values, next_value, chunk_cues, is_next_iti, gamma=0.99):
    returns = torch.zeros_like(rewards)
    
    # 1. Sever the chunk boundary if the next chunk starts with an ITI
    running_return = next_value.detach() * (1.0 - is_next_iti)
    
    is_iti = (chunk_cues == -1).float()
    
    # Iterate backwards through time to calculate discounted returns
    for t in reversed(range(rewards.size(1))):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        
        # 2. BIOLOGICAL SEVERING: Wipe the return for the NEXT backward step 
        # if the CURRENT step is an ITI. This completely isolates trials.
        running_return = running_return * (1.0 - is_iti[:, t])
        
    return returns

def train_rl_model(
        total_timesteps=100000,
        bptt_horizon =60, 
        batch_size=32, 
        lr=1e-3, 
        gamma=0.99, 
        entropy_coef=0.01,
        save_dir="checkpoints", 
        critic_coef=2,
        alpha_plus=1.0,
        alpha_minus=1.0):
    
    # Initialize environment and model
    env = ContinuousRLTask(
        batch_size=batch_size, 
        total_timesteps=total_timesteps,
        stimulus_duration=TASK.get("stimulus_duration", 10),
        delay_duration=TASK.get("delay_duration", 0),
        reward_duration=TASK.get("reward_duration", 5)
    )
    model = ActorCriticRNN(input_size=3, hidden_size=64, num_actions=2)

    
    optimizer = optim.Adam(model.parameters(), lr=lr)    

    loss_history = []
    live_performance = {'A': [], 'B': [], 'C': []}
    live_actor_performance = {'A': [], 'B': [], 'C': []}
    h_t = None # Initialize hidden state outside the loop so it can persist across batches (truncated BPTT)
    
    
    timesteps_per_epoch = 1000 
    current_epoch = 0
    epoch_vals = {'A': [], 'B': [], 'C': []}    
    epoch_actor_vals = {'A': [], 'B': [], 'C': []}
    
    
    # Force the reversal timestep to perfectly match the target epoch
    reversal_epoch = TRAINING.get("reversal_epoch", 200)
    reversal_timestep = reversal_epoch * timesteps_per_epoch
    print(f"Starting Continuous Training: {total_timesteps} total steps, chunked by {bptt_horizon}.")

    for t_start in range(0, total_timesteps, bptt_horizon):
        if t_start % timesteps_per_epoch == 0:
            h_t = None # Reset hidden state at epoch boundaries to prevent unintended carryover
        t_end = min(t_start + bptt_horizon, total_timesteps)

        # Reversal
        if t_start <= reversal_timestep < t_end: 
            print(f"\n--- REVERSING STIMULUS VALUES at timestep {reversal_timestep} ---")

            # 1. Stimulus A (index 0) goes from 100% -> 0% reward
            stim_a_mask = (env.trial_cues[:, reversal_timestep:] == 0)
            env.trial_is_rewarded[:, reversal_timestep:][stim_a_mask] = False

            # 2. Stimulus C (index 2) goes from 0% -> 100% reward
            stim_c_mask = (env.trial_cues[:, reversal_timestep:] == 2)
            env.trial_is_rewarded[:, reversal_timestep:][stim_c_mask] = True
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr # Reset LR at reversal
        
        chunk_inputs = env.inputs[:, t_start:t_end, :]
        #Forward pass through the model for this chunk
        action_probs, values, h_t = model(chunk_inputs, h_0=h_t) # Forward pass through the model for this chunk

        h_t = h_t.detach() # Detach hidden state to prevent backprop through entire history (truncated BPTT)

        # Calculate action log probabilities and sample actions
        dist = Categorical(action_probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        #evaluate the rewards for the actions taken
        rewards = env.evaluate_chunk(t_start, t_end, actions)

        chunk_cues = env.trial_cues[:, t_start:t_end]
        chunk_reward_windows = env.reward_windows[:, t_start:t_end]

        #bootstrap the value of the next state using the critic's estimate (for continuous returns)
        if t_end < total_timesteps:
            with torch.no_grad():
                next_input = env.inputs[:, t_end:t_end + 1, :]
                _, next_val, _ = model(next_input, h_0=h_t)
                next_value_scalar = next_val.squeeze(1)

                next_cues = env.trial_cues[:, t_end]
                is_next_iti = (next_cues == -1).float()
        else:
            next_value_scalar = torch.zeros(batch_size)
            is_next_iti = torch.ones(batch_size) # If we're at the end, treat the "next" state as an ITI to sever returns

        returns = compute_returns_continuous(rewards, values, next_value_scalar, chunk_cues, is_next_iti, gamma)
            
        #Calculate losses
        advantage = returns - values.detach()
        

        #Inducing disease - split RPE modulation
        positive_rpe = torch.clamp(advantage, min=0.0) * alpha_plus
        negative_rpe = torch.clamp(advantage, max=0.0) * alpha_minus

        modulated_advantage = positive_rpe + negative_rpe

        actor_loss = -(log_probs * modulated_advantage).mean() # Policy gradient loss (encourage actions that had positive advantage)
        critic_target = values.detach() + modulated_advantage
        
        critic_loss = F.mse_loss(values, critic_target) # Critic loss (fit value estimates to the computed returns)
        entropy = dist.entropy().mean() # Entropy bonus (encourage exploration)

        total_loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

        # Backpropagation and optimization step
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(total_loss.item())

        # Isolate the Critic's predictions specifically during the reward windows
        for stim_idx, cue_name in enumerate(['A', 'B', 'C']):
            # Use the logical NOT operator (~) on the reward window
            mask = chunk_inputs[..., stim_idx] > 0.5
            actor_mask = (chunk_reward_windows == True) & (chunk_cues == stim_idx)
            if mask.any():
                epoch_vals[cue_name].append(values[mask].mean().item())
                epoch_actor_vals[cue_name].append(action_probs[:, :, 1][actor_mask].mean().item())
        
        #VIRTUAL EPOCH BOUNDARIES for tracking performance and saving checkpoints (since we're not training in discrete epochs, we define our own "epochs" based on timesteps)
        if (t_start + bptt_horizon) >= (current_epoch + 1) * timesteps_per_epoch:
            
            #Average the values for this epoch (fallback to previous if no trials occurred)
            mean_A = np.mean(epoch_vals['A']) if epoch_vals['A'] else (live_performance['A'][-1] if live_performance['A'] else 0.0)
            mean_B = np.mean(epoch_vals['B']) if epoch_vals['B'] else (live_performance['B'][-1] if live_performance['B'] else 0.0)
            mean_C = np.mean(epoch_vals['C']) if epoch_vals['C'] else (live_performance['C'][-1] if live_performance['C'] else 0.0)
            
            live_performance['A'].append(mean_A)
            live_performance['B'].append(mean_B)
            live_performance['C'].append(mean_C)

            act_A = np.mean(epoch_actor_vals['A']) if epoch_actor_vals['A'] else (live_actor_performance['A'][-1] if live_actor_performance['A'] else 0.0)
            act_B = np.mean(epoch_actor_vals['B']) if epoch_actor_vals['B'] else (live_actor_performance['B'][-1] if live_actor_performance['B'] else 0.0)
            act_C = np.mean(epoch_actor_vals['C']) if epoch_actor_vals['C'] else (live_actor_performance['C'][-1] if live_actor_performance['C'] else 0.0)
            
            live_actor_performance['A'].append(act_A)
            live_actor_performance['B'].append(act_B)
            live_actor_performance['C'].append(act_C)
            
            # Reset accumulators for the next epoch
            epoch_vals = {'A': [], 'B': [], 'C': []}
            epoch_actor_vals = {'A': [], 'B': [], 'C': []}
            
            # Save Checkpoints exactly as plotting.py expects
            os.makedirs(save_dir, exist_ok=True)
            if current_epoch % 2 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"weights_epoch_{current_epoch}.pth"))
            
 
            reversal_epoch = TRAINING.get("reversal_epoch", 200)
            if current_epoch == (reversal_epoch - 1):
                torch.save(model.state_dict(), os.path.join(save_dir, "weights_baseline.pth"))
            elif current_epoch == (reversal_epoch + 5):
                torch.save(model.state_dict(), os.path.join(save_dir, "weights_early_reversal.pth"))
            elif current_epoch == (TRAINING.get("epochs", 400) - 1):
                torch.save(model.state_dict(), os.path.join(save_dir, "weights_final_reversal.pth"))
            # Print exactly 10 epochs worth of performance data at a time to avoid overwhelming the console
            if (current_epoch + 1) % 10 == 0:
                total_epochs = total_timesteps // timesteps_per_epoch
                
                print(f"[Virtual Epoch {(current_epoch+1):03d}/{total_epochs:03d}] "
                      f"Timestep {t_end:06d}/{total_timesteps:06d} "
                      f"| Loss: {total_loss.item():.4f} "
                      f"| Licks -> A: {mean_A:.2f}, B: {mean_B:.2f}, C: {mean_C:.2f}")
            current_epoch += 1
                    # Specific milestone saves for the baseline/reversal plots

    # Save final states
    
    torch.save(live_performance, os.path.join(save_dir, "live_performance.pt"))
    
    return model, loss_history, live_performance, live_actor_performance

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

