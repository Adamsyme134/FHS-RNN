from task import *
from models import *
from train import *
import matplotlib.pyplot as plt
import numpy as np

lr = TRAINING["lr"]
batch_size = TRAINING["batch_size"]
epochs = TRAINING["epochs"]
batches_per_epoch = TRAINING["batches_per_epoch"]


def plot_task_batch():
    padded_inputs, padded_targets, lengths, mask = generate_batch(5)
    print(padded_inputs, padded_targets, lengths, mask)
    fig, axes = plt.subplots(5, 1, figsize=(10, 8))

    for i in range(5):

        # Plot input matrix
        axes[i].imshow(
            padded_inputs[i].numpy().T,
            aspect='auto',
            cmap='Greys',
            interpolation='nearest'
        )

        # Get target sequence
        target = padded_targets[i].numpy()

        # Time axis
        x = np.arange(len(target))

        # Map lick rate values onto vertical positions
        # 0 -> bottom, 1 -> top
        y = 3 - (target * 3)

        # Overlay in red
        axes[i].plot(
            x,
            y,
            color='red',
            linewidth=2
        )

        axes[i].set_yticks([0,1,2,3])
        axes[i].set_yticklabels(['A','B','C','R'])

    plt.tight_layout()
    plt.show()

def plot_predictions(rnn,batch_size =4):
    
    with torch.no_grad(): #Do not track gradients 
        #Make fresh batch for testing
        inputs, targets, lengths, mask = generate_batch(4,["A","B","B","C"],[1,1,0,0])

        ys, hs = rnn.forward(inputs)

        #Convert to numpy for plotting 
        ys_np = ys.detach().numpy()
        targets_np = targets.detach().numpy()    
    

    fig, axes = plt.subplots(batch_size, 1, figsize=(10, 2 * batch_size), sharex=True)
    for i in range(batch_size):
        trial_len = lengths[i].item() # Actual unpadded length

        # Plot input matrix (sliced to actual length)
        axes[i].imshow(
            inputs[i, :trial_len].detach().numpy().T,
            aspect='auto',
            cmap='Greys',
            interpolation='nearest'
        )
        target_seq = targets[i, :trial_len, 0].detach().numpy()
        pred_seq = ys[i, :trial_len, 0].detach().numpy()
        
        y_target = 3 - (target_seq * 3)
        y_pred = 3 - (pred_seq * 3)

        x_axis = np.arange(trial_len) #ensure lines stop when trial ends

        #Overlay the lines for target and moel prediction
        axes[i].plot(x_axis, y_target, color='red', label='Target', linewidth=2)
        axes[i].plot(x_axis, y_pred, color='green', label='RNN Prediction', linestyle='--')

        #Labels and formatting
        axes[i].set_yticks([0, 1, 2, 3])
        axes[i].set_yticklabels(['A', 'B', 'C', 'R'])
        axes[i].set_ylabel(f"Trial {i}")

    axes[-1].set_xlabel("Timestep")
    # Only show legend on the first subplot
    axes[0].legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

rnn, loss_history = train_model(ScratchRNN(), lr, epochs, batches_per_epoch, batch_size)

plot_predictions(rnn)