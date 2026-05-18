from task import *
from models import *
from train import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

lr = TRAINING["lr"]
batch_size = TRAINING["batch_size"]
epochs = TRAINING["epochs"]
batches_per_epoch = TRAINING["batches_per_epoch"]


def plot_task_batch():
    padded_inputs, padded_targets, lengths, mask = generate_batch(batch_size=5)
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

def plot_predictions(rnn,batch_size =4,is_reversed=False):
    
    with torch.no_grad(): #Do not track gradients 
        #Make fresh batch for testing
        inputs, targets, lengths, mask = generate_batch(
            is_reversed=is_reversed,
            cues=["A","B","B","C"],
            rewards=[1,1,0,0] if not is_reversed else [0,0,1,1])

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

def extract_hidden_states(rnn, num_trials_per_stim=50, is_reversed=False):
    #Generates a structured batch and extracts hidden states.
    rnn.eval()
    
    # Create an equal number of A, B, and C trials
    cues = ["A"] * num_trials_per_stim + ["B"] * num_trials_per_stim + ["C"] * num_trials_per_stim
    
    with torch.no_grad():
        inputs, targets, lengths, mask = generate_batch(
            is_reversed=is_reversed,
            batch_size=len(cues),
            cues=cues)
        
        # Run forward pass
        ys, hs = rnn(inputs)
        
    return hs, cues, lengths

def plot_pca_trajectories(hs, cues, lengths, is_reversed = False):
    #Fits PCA to hidden states and plots 2D trajectories
    # Convert to numpy
    hs_np = hs.detach().numpy()
    batch_size, max_seq_len, hidden_size = hs_np.shape
    
    # 1. Isolate valid hidden states (drop padding)
    valid_hs = []
    for i in range(batch_size):
        trial_len = lengths[i].item()
        valid_hs.append(hs_np[i, :trial_len, :])
        
    # 2. Flatten and fit PCA
    # Concatenate along the timestep axis to create shape: (total_valid_timesteps, hidden_size)
    flat_hs = np.concatenate(valid_hs, axis=0) 
    
    pca = PCA(n_components=2)
    pca.fit(flat_hs)
    
    # 3. Plotting setup
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'A': 'blue', 'B': 'orange', 'C': 'green'}
    
    # 4. Transform and plot each trial
    for i in range(batch_size):
        trial_len = lengths[i].item()
        trial_hs = hs_np[i, :trial_len, :]
        
        # Project this specific trial into 2D space
        trial_pca = pca.transform(trial_hs)
        
        # Plot the trajectory (alpha=0.3 makes overlapping lines visible)
        ax.plot(trial_pca[:, 0], trial_pca[:, 1], color=colors[cues[i]], alpha=0.3)
        
        # Mark the start of the trial (ITI) with a small black dot
        ax.scatter(trial_pca[0, 0], trial_pca[0, 1], color='black', s=10)
        
        # Mark the end of the trial (Reward window) with an X
        ax.scatter(trial_pca[-1, 0], trial_pca[-1, 1], color=colors[cues[i]], s=30, marker='x')

    # Formatting
    plt.title("PCA of RNN Hidden State Trajectories")
    
    # Show how much variance the components actually capture
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    plt.xlabel(f"Principal Component 1 ({pc1_var:.1f}% variance)")
    plt.ylabel(f"Principal Component 2 ({pc2_var:.1f}% variance)")
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='orange', lw=2),
        Line2D([0], [0], color='green', lw=2)
    ]

    if not is_reversed:
        labels = ['Stimulus A (100%)', 'Stimulus B (50%)', 'Stimulus C (0%)']
    else:
        labels = ['Stimulus A (0%)', 'Stimulus B (50%)', 'Stimulus C (100%)']    
    ax.legend(custom_lines, labels)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_clusters(hs, cues, lengths, is_reversed=False):
    """Fits PCA to a single snapshot in time (end of delay) to show state clusters."""
    # Convert to numpy
    hs_np = hs.detach().numpy()
    batch_size = hs_np.shape[0]
    
    #Isolate the specific timestep (Snapshot)
    snapshot_hs = []
    for i in range(batch_size):
        trial_len = lengths[i].item()
        
        #get the timestep right BEFORE the reward is delivered.
        anticipation_timestep = trial_len - 2 
        snapshot_hs.append(hs_np[i, anticipation_timestep, :])
        
    # onvert list of 1D arrays into a 2D array of shape (batch_size, hidden_size)
    snapshot_matrix = np.array(snapshot_hs) 
    
    #Fit PCA on ONLY these snapshots
    pca = PCA(n_components=2)
    snapshot_pca = pca.fit_transform(snapshot_matrix)
    
    #Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'A': 'blue', 'B': 'orange', 'C': 'green'}
    
    for i in range(batch_size):
        ax.scatter(
            snapshot_pca[i, 0], 
            snapshot_pca[i, 1], 
            color=colors[cues[i]], 
            alpha=0.7,
            s=60 # Make dots slightly larger for visibility
        )

    # Formatting
    reversed_title = "Normal inputs" if not is_reversed else "Reversed inputs"
    plt.title("Representational Snapshot (End of Delay Period) | "+reversed_title)
    
    # Show how much variance the components actually capture for these snapshots
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    plt.xlabel(f"Principal Component 1 ({pc1_var:.1f}% variance)")
    plt.ylabel(f"Principal Component 2 ({pc2_var:.1f}% variance)")
    
    #Legend (using dots instead of lines)
    from matplotlib.lines import Line2D
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)
    ]

    if not is_reversed:
        labels = ['Stimulus A (100%)', 'Stimulus B (50%)', 'Stimulus C (0%)']
    else:
        labels = ['Stimulus A (0%)', 'Stimulus B (50%)', 'Stimulus C (100%)']
        
    ax.legend(custom_legend, labels)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_distance_matrix(hs, cues, lengths, is_reversed=False):
    #Calculate the euclidean distance between representations of stimuli
    hs_np = hs.detach().numpy()
    batch_size = hs_np.shape[0]

    snapshots = {'A': [], 'B': [], 'C': []} 
    #Get the same snapshots as in the cluster plot
    for i in range(batch_size):
        trial_len = lengths[i].item()
        anticipation_timestep = trial_len - 2 #get the same timestep just before the reward

        cue = cues[i]
        snapshots[cue].append(hs_np[i, anticipation_timestep, :])
    
    #Calculate the mean state (centroid) for each stimulus in the full dimensional space
    centroids = {
        'A': np.mean(snapshots['A'], axis=0),
        'B': np.mean(snapshots['B'], axis=0),
        'C': np.mean(snapshots['C'], axis=0)
    }    

    
    if not is_reversed:
        labels = ['Stimulus A (100%)', 'Stimulus B (50%)', 'Stimulus C (0%)']
    else:
        labels = ['Stimulus A (0%)', 'Stimulus B (50%)', 'Stimulus C (100%)']
    keys = ['A', 'B', 'C']
    dist_matrix = np.zeros((3, 3)) 
    #Calculate euclidean distances
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            dist_matrix[i,j] = np.linalg.norm(centroids[k1] - centroids[k2]) #Distance formula

    #Plot the heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(dist_matrix, cmap='viridis')
    fig.colorbar(cax, label='Euclidean Distance')     

    #Label squares with exact values
    for i in range(3):
        for j in range(3):
            text_color = "black" if dist_matrix[i, j] > (np.max(dist_matrix)/2) else "white"
            ax.text(j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center", color=text_color)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.title("Representational Distance Matrix")
    plt.show()    


#Train the model
#train_model(ScratchRNN(), lr, epochs, batches_per_epoch, batch_size)
#Plotting the baseline (pre-reversal)
rnn = ScratchRNN()
rnn.load_state_dict(torch.load("weights_baseline.pth")) #load pretrained model

hs_base, cues_base, lengths_base = extract_hidden_states(rnn, num_trials_per_stim=100, is_reversed=False)
plot_clusters(hs_base, cues_base, lengths_base, is_reversed=False)
plot_distance_matrix(hs_base, cues_base, lengths_base, is_reversed=False)

#Plot the fully reversed model
rnn_rev = ScratchRNN()
rnn_rev.load_state_dict(torch.load("weights_final_reversal.pth"))

hs_rev, cues_rev, lengths_rev = extract_hidden_states(rnn_rev, num_trials_per_stim=100, is_reversed=True)
plot_clusters(hs_rev, cues_rev, lengths_rev, is_reversed=True)
plot_distance_matrix(hs_rev, cues_rev, lengths_rev, is_reversed=True)