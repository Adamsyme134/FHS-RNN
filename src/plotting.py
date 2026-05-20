from task import *
from models import *
from train import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import ffmpeg
import glob
import re



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
    
    return hs, cues, lengths,inputs

def plot_pca_trajectories(hs, cues, lengths, inputs, is_reversed = False):
    #Fits PCA to hidden states and plots 2D trajectories


    colors = {
        'A': ['#086ec7', '#17dae8', '#74029e', '#a032a8'],
        'B': ['#f5f500', '#f5a700', '#f50c00', '#5e0909'],
        'C': ['#E5F5E0', '#A1D99B', '#31A354', '#006D2C']
    } #Colours to split each stimulus into four sections (ITI,cue,delay,reward)

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
    
    # 4. Transform and plot each trial
    for i in range(batch_size):
        trial_len = lengths[i].item()
        trial_hs = hs_np[i, :trial_len, :]
        cue=cues[i]

        trial_inputs = inputs[i, :trial_len].detach().numpy()
        stim_activity = np.sum(trial_inputs[:, 0:3], axis=1)
        cue_active_timesteps = np.where(stim_activity > 0)[0] #Find the cue period

        if len(cue_active_timesteps) > 0:
            # The cue starts at the first active timestep
            t_cue_start = cue_active_timesteps[0]
    
            # The delay starts on the timestep immediately after the cue turns off
            t_delay_start = cue_active_timesteps[-1] + 1
        else:
            # Fallback just in case
            print("Error with trial plotting")
            t_cue_start = 3 
            t_delay_start = 8    

        t_reward_start = trial_len -2 #Reward timestep
        trial_pca = pca.transform(trial_hs)

        # Phase 1: ITI
        ax.plot(trial_pca[:t_cue_start+1, 0], trial_pca[:t_cue_start+1, 1], 
                color=colors[cue][0], alpha=0.2)
                
        # Phase 2: Cue 
        ax.plot(trial_pca[t_cue_start:t_delay_start+1, 0], trial_pca[t_cue_start:t_delay_start+1, 1], 
                color=colors[cue][1], alpha=0.2)
                
        # Phase 3: Delay
        ax.plot(trial_pca[t_delay_start:t_reward_start+1, 0], trial_pca[t_delay_start:t_reward_start+1, 1], 
                color=colors[cue][2], alpha=0.2)
                
        # Phase 4: Reward 
        ax.plot(trial_pca[t_reward_start:, 0], trial_pca[t_reward_start:, 1], 
                color=colors[cue][3], alpha=0.2)

        # Mark the start of the trial (ITI) with a small black dot
        ax.scatter(trial_pca[0, 0], trial_pca[0, 1], color='black', s=10)
        
        # Mark the end of the trial (Reward window) with an X
        ax.scatter(trial_pca[-1, 0], trial_pca[-1, 1], color=colors[cue][3], s=30, marker='x')

    # Formatting
    plt.title("PCA of RNN Hidden State Trajectories")
    
    # Show how much variance the components actually capture
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    plt.xlabel(f"Principal Component 1 ({pc1_var:.1f}% variance)")
    plt.ylabel(f"Principal Component 2 ({pc2_var:.1f}% variance)")
    
    # Legend
    phases = ['ITI', 'Cue', 'Delay', 'Reward']
    
    if not is_reversed:
        stim_names = {'A': 'Stim A (100%)', 'B': 'Stim B (50%)', 'C': 'Stim C (0%)'}
    else:
        stim_names = {'A': 'Stim A (0%)', 'B': 'Stim B (50%)', 'C': 'Stim C (100%)'}
    custom_lines = []
    custom_labels = []
    for stim in ['A', 'B', 'C']:
        for i, phase in enumerate(phases):
            # Grab the specific hex code for this stimulus and phase
            custom_lines.append(Line2D([0], [0], color=colors[stim][i], lw=3))
            custom_labels.append(f"{stim_names[stim]} - {phase}")

    ax.legend(custom_lines, custom_labels, loc='lower center', 
              bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
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

def animate_pca_trajectories(weight_paths, is_reversed=False, num_trials_per_stim=10):
    # 1. Initialize network and generate fixed input data
    # Use the exact same trials for every epoch to cleanly see learning
    rnn = ScratchRNN()
    cues = ["A"] * num_trials_per_stim + ["B"] * num_trials_per_stim + ["C"] * num_trials_per_stim
    
    with torch.no_grad():
        inputs, targets, lengths, mask = generate_batch(
            is_reversed=is_reversed,
            batch_size=len(cues),
            cues=cues
        )
        
    # 2. Extract hidden states for ALL checkpoints using the exact same inputs
    all_epochs_hs = []
    print("Extracting hidden states across all checkpoints...")
    for path in weight_paths:
        rnn.load_state_dict(torch.load(path))
        rnn.eval()
        with torch.no_grad():
            _, hs = rnn(inputs)
        all_epochs_hs.append(hs.detach().numpy())
        
    # 3. Fit PCA space on the FINAL checkpoint to ensure coordinate stability
    final_hs_np = all_epochs_hs[-1]
    batch_size = len(cues)
    valid_hs_final = []
    for i in range(batch_size):
        trial_len = lengths[i].item()
        valid_hs_final.append(final_hs_np[i, :trial_len, :])
    
    flat_hs_final = np.concatenate(valid_hs_final, axis=0)
    pca = PCA(n_components=2)
    pca.fit(flat_hs_final)
    
    # 4. Pre-transform all epochs into the final PCA space and find global limits
    all_epochs_pca = []
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    for epoch_hs in all_epochs_hs:
        epoch_pca = []
        for i in range(batch_size):
            trial_len = lengths[i].item()
            trial_hs = epoch_hs[i, :trial_len, :]
            transformed = pca.transform(trial_hs)
            epoch_pca.append(transformed)
            
            # Update global bounds so the camera stays still
            x_min = min(x_min, transformed[:,0].min())
            x_max = max(x_max, transformed[:,0].max())
            y_min = min(y_min, transformed[:,1].min())
            y_max = max(y_max, transformed[:,1].max())
            
        all_epochs_pca.append(epoch_pca)
        
    # Add padding to limits
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    # 5. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'A': ['#086ec7', '#17dae8', '#74029e', '#a032a8'],
        'B': ['#f5f500', '#f5a700', '#f50c00', '#5e0909'],
        'C': ['#E5F5E0', '#A1D99B', '#31A354', '#006D2C']
    }
    
    phases = ['ITI', 'Cue', 'Delay', 'Reward']
    stim_names = {'A': 'Stim A (100%)', 'B': 'Stim B (50%)', 'C': 'Stim C (0%)'} if not is_reversed else {'A': 'Stim A (0%)', 'B': 'Stim B (50%)', 'C': 'Stim C (100%)'}
    
    # Pre-build legend items
    custom_lines, custom_labels = [], []
    for stim in ['A', 'B', 'C']:
        for i, phase in enumerate(phases):
            custom_lines.append(Line2D([0], [0], color=colors[stim][i], lw=3))
            custom_labels.append(f"{stim_names[stim]} - {phase}")

    def update(frame_idx):
        ax.clear()
        
        # Lock axes
        ax.set_xlim([x_min - x_pad, x_max + x_pad])
        ax.set_ylim([y_min - y_pad, y_max + y_pad])
        
        ax.set_title(f"PCA Trajectory Evolution | Epoch Checkpoint: {frame_idx + 1}/{len(weight_paths)}")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
        frame_data = all_epochs_pca[frame_idx]
        
        for i in range(batch_size):
            trial_len = lengths[i].item()
            cue = cues[i]
            
            trial_inputs = inputs[i, :trial_len].detach().numpy()
            stim_activity = np.sum(trial_inputs[:, 0:3], axis=1)
            cue_active = np.where(stim_activity > 0)[0]
            
            if len(cue_active) > 0:
                t_cue_start = cue_active[0]
                t_delay_start = cue_active[-1] + 1
            else:
                t_cue_start = 3 
                t_delay_start = 8   
                
            t_reward_start = trial_len - 2
            trial_pca = frame_data[i]
            
            # Plot segments
            ax.plot(trial_pca[:t_cue_start+1, 0], trial_pca[:t_cue_start+1, 1], color=colors[cue][0], alpha=0.3)
            ax.plot(trial_pca[t_cue_start:t_delay_start+1, 0], trial_pca[t_cue_start:t_delay_start+1, 1], color=colors[cue][1], alpha=0.3)
            ax.plot(trial_pca[t_delay_start:t_reward_start+1, 0], trial_pca[t_delay_start:t_reward_start+1, 1], color=colors[cue][2], alpha=0.3)
            ax.plot(trial_pca[t_reward_start:, 0], trial_pca[t_reward_start:, 1], color=colors[cue][3], alpha=0.3)
            
            # Markers
            ax.scatter(trial_pca[0, 0], trial_pca[0, 1], color='black', s=10)
            ax.scatter(trial_pca[-1, 0], trial_pca[-1, 1], color=colors[cue][3], s=30, marker='x')

        # Re-add legend since ax.clear() wipes it
        ax.legend(custom_lines, custom_labels, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Generate animation
    ani = FuncAnimation(fig, update, frames=len(weight_paths), interval=150, repeat=False)
    plt.tight_layout()
    return ani

def extract_epoch(filename):
    match = re.search(r'weights_epoch_(\d+)', filename)
    return int(match.group(1)) if match else -1

def plot_performance_timeline(weight_paths, reversal_epoch, num_trials_per_stim=20):
    """
    Plots the model's predicted lick rate for Stimulus A, B, and C across training epochs.
    """
    rnn = ScratchRNN()
    
    # Generate a fixed test set of physical inputs (is_reversed=False)
    # We just want to see how the network reacts to the physical cues over time
    cues = ["A"] * num_trials_per_stim + ["B"] * num_trials_per_stim + ["C"] * num_trials_per_stim
    
    with torch.no_grad():
        inputs, _, lengths, _ = generate_batch(is_reversed=False, batch_size=len(cues), cues=cues)
    
    epochs = []
    preds_A = []
    preds_B = []
    preds_C = []
    
    print("Evaluating checkpoints for performance timeline...")
    
    # Filter out baseline/final names if they don't have an epoch number 
    # to keep the X-axis mathematically clean, relying purely on the epoch intervals
    numbered_paths = [p for p in weight_paths if extract_epoch(p) != -1]
    
    for path in numbered_paths:
        epoch = extract_epoch(path)
            
        rnn.load_state_dict(torch.load(path))
        rnn.eval()
        
        with torch.no_grad():
            ys, _ = rnn(inputs)
            
        ys_np = ys.detach().numpy()
        
        batch_preds = {'A': [], 'B': [], 'C': []}
        for i in range(len(cues)):
            cue = cues[i]
            trial_len = lengths[i].item()
            
            # Isolate the prediction at the exact anticipation timestep
            anticipation_timestep = trial_len - 2
            pred_val = ys_np[i, anticipation_timestep, 0] 
            batch_preds[cue].append(pred_val)
            
        epochs.append(epoch)
        preds_A.append(np.mean(batch_preds['A']))
        preds_B.append(np.mean(batch_preds['B']))
        preds_C.append(np.mean(batch_preds['C']))
        
    # Sort chronologically
    sort_idx = np.argsort(epochs)
    epochs = np.array(epochs)[sort_idx]
    preds_A = np.array(preds_A)[sort_idx]
    preds_B = np.array(preds_B)[sort_idx]
    preds_C = np.array(preds_C)[sort_idx]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, preds_A, label='Stimulus A', color='blue', linewidth=2.5)
    plt.plot(epochs, preds_B, label='Stimulus B (Constant 50%)', color='orange', linewidth=2.5)
    plt.plot(epochs, preds_C, label='Stimulus C', color='green', linewidth=2.5)
    
    # Mark Reversal Point
    plt.axvline(x=reversal_epoch, color='black', linestyle='--', linewidth=2, label='Reversal Initiated')
    
    # Target Guidelines
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.6)
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6)
    plt.axhline(y=0.0, color='gray', linestyle=':', alpha=0.6)
    
    # Formatting
    plt.title('Evolution of Predicted Lick Rates During Anticipation Window', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Predicted Lick Rate', fontsize=12)
    plt.ylim(-0.1, 1.1)
    
    # Annotate Phases
    mid_acq = reversal_epoch / 2
    mid_rev = reversal_epoch + (epochs[-1] - reversal_epoch) / 2
    plt.text(mid_acq, 1.05, "Acquisition\n(A=100%, C=0%)", ha='center', fontsize=10, fontweight='bold')
    plt.text(mid_rev, 1.05, "Reversal\n(A=0%, C=100%)", ha='center', fontsize=10, fontweight='bold')

    plt.legend(loc='center right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_all_graphs(train=False):
    lr = TRAINING["lr"]
    batch_size = TRAINING["batch_size"]
    epochs = TRAINING["epochs"]
    batches_per_epoch = TRAINING["batches_per_epoch"]
    checkpoint_files = glob.glob("weights_epoch_*.pth")
    checkpoint_files.sort(key=extract_epoch)

    #Train the model
    if train:
        rnn, _ = train_model(ScratchRNN(), lr, epochs, batches_per_epoch, batch_size)

    checkpoint_files = glob.glob("checkpoints/weights_epoch_*.pth")
    checkpoint_files.sort(key=extract_epoch)

    REVERSAL_EPOCH = TRAINING["reversal_epoch"]

    if "checkpoints/weights_baseline.pth" in glob.glob("checkpoints/*.pth"):
        checkpoint_files.append("checkpoints/weights_baseline.pth") #Final  output
    
    print(f"Found {len(checkpoint_files)} checkpoints to animate.")

    plot_performance_timeline(checkpoint_files, reversal_epoch=TRAINING["reversal_epoch"])
    # 2. Run the animation 
    acquisition_files = [f for f in checkpoint_files if extract_epoch(f) < REVERSAL_EPOCH]
    acquisition_files.append("checkpoints/weights_baseline.pth") # Chronologically belongs at the end here

    print(f"Animating Acquisition: {len(acquisition_files)} frames...")
    # is_reversed=False because it is learning the normal task
    anim_acq = animate_pca_trajectories(acquisition_files, is_reversed=False, num_trials_per_stim=10)
    anim_acq.save('pca_acquisition.mp4', writer='ffmpeg', fps=10)
    print("Saved pca_acquisition.mp4!")

    #REVERSAL ANIMATION
    # Start with the baseline model, then add all epochs after reversal
    reversal_files = ["checkpoints/weights_baseline.pth"] + [f for f in checkpoint_files if extract_epoch(f) >= REVERSAL_EPOCH]

    print(f"Animating Reversal: {len(reversal_files)} frames...")
    #is_reversed=True so the network gets the swapped inputs it is actually learning
    anim_rev = animate_pca_trajectories(reversal_files, is_reversed=True, num_trials_per_stim=10)
    anim_rev.save('pca_reversal.mp4', writer='ffmpeg', fps=10)
    print("Saved pca_reversal.mp4!")

    #Plotting the baseline (pre-reversal)
    rnn = ScratchRNN()

    rnn.load_state_dict(torch.load("checkpoints/weights_baseline.pth")) #load pretrained model
    
    hs_base, cues_base, lengths_base, inputs_base = extract_hidden_states(rnn, num_trials_per_stim=100, is_reversed=False)
    plot_pca_trajectories(hs_base, cues_base, lengths_base, is_reversed=False, inputs=inputs_base)
    plot_clusters(hs_base, cues_base, lengths_base, is_reversed=False)
    plot_distance_matrix(hs_base, cues_base, lengths_base, is_reversed=False)

    #Plot the fully reversed model
    rnn_rev = ScratchRNN()
    rnn_rev.load_state_dict(torch.load("checkpoints/weights_final_reversal.pth"))

    hs_rev, cues_rev, lengths_rev,inputs_rev = extract_hidden_states(rnn_rev, num_trials_per_stim=100, is_reversed=True)
    plot_pca_trajectories(hs_rev, cues_rev, lengths_rev, is_reversed=True, inputs=inputs_rev)
    plot_clusters(hs_rev, cues_rev, lengths_rev, is_reversed=True)
    plot_distance_matrix(hs_rev, cues_rev, lengths_rev, is_reversed=True)
    plot_predictions(rnn_rev, is_reversed=True)


plot_all_graphs(False)