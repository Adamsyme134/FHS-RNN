from models import ScratchRNN
from task import *
import numpy as np
from configs import TRAINING

lr = TRAINING["lr"]
batch_size = TRAINING["batch_size"]
epochs = TRAINING["epochs"]
batches_per_epoch = TRAINING["batches_per_epoch"]


rnn = ScratchRNN()

def train_model(rnn, lr, epochs, batches_per_epoch, batch_size):
    loss_history = []
    for e in range (epochs):
        epoch_loss = 0
        for b in range (batches_per_epoch): #eachx batch will update the parameters once
            padded_inputs, padded_targets, lengths, mask = generate_batch(batch_size)
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
        avg_loss /= (batches_per_epoch)
        loss_history.append(avg_loss)
        print(f"Epoch: {e} | Loss: {avg_loss:.4f}")
    
    return rnn, loss_history #returns a fully trained RNN