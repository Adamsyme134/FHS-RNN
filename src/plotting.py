from xml.parsers.expat import model

from task import *
from models import *
from train import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import scipy.stats as stats
from matplotlib.animation import FuncAnimation
from evaluate import *
from dataclasses import dataclass
from pathlib import Path
import ffmpeg
import glob
import re
import torch
import datetime
import uuid
import json
import seaborn as sns
from evaluate import *

#Configurations for plotting
@dataclass
class PlotConfig:
    lr: float
    batch_size: int
    epochs: int
    batches_per_epoch: int
    reversal_epoch: int
    num_trials_per_stim: int = 10
    fps: int = 10

#----- SPECIFIC PLOTS ----
def plot_task_batch():
    padded_inputs, padded_targets, lengths, mask, cues = generate_batch(batch_size=5)
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

def plot_learning_curves(df_results):
    plt.figure(figsize=(10, 6))
    
    # Loop through every run in the dataframe
    for index, row in df_results.iterrows():
        # Create a label showing the hyperparameters for this line
        label = f"LR:{row['learning_rate']}, HS:{row['hidden_size']}, Sig:{row['target_sigma']}"
        plt.plot(row['loss_history'], label=label, alpha=0.7)
        
    plt.title('Learning Curves across Hyperparameters')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # Place legend outside the plot so it doesn't cover the lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_heatmap(df_results, param1='hidden_size', param2='target_sigma'):
    plt.figure(figsize=(8, 6))
    
    # Aggregate the data. If there are multiple Learning Rates, this takes the mean final_loss
    pivot_table = df_results.pivot_table(
        values='final_loss', 
        index=param1, 
        columns=param2, 
        aggfunc='mean'
    )
    
    # Use seaborn for a clean heatmap
    sns.heatmap(pivot_table, annot=True, cmap='viridis_r', fmt=".4f")
    plt.title(f'Final Loss: {param1} vs {param2}')
    plt.tight_layout()
    plt.show()

def plot_rl_convergence_heatmap(df_results, param1='gamma', param2='entropy_coef'):
    """
    Plots a heatmap showing how fast the model converges based on RL parameters.
    """
    plt.figure(figsize=(8, 6))
    
    # Filter out runs that didn't converge for the plot, or set them to a high max value
    df_plot = df_results.copy()
    max_epoch = df_plot[df_plot['convergence_epoch'] != "Did Not Converge"]['convergence_epoch'].max()
    
    # Replace "Did Not Converge" with max_epoch + 20% so they show up as the "worst" color
    df_plot['convergence_epoch'] = df_plot['convergence_epoch'].replace(
        "Did Not Converge", int(max_epoch * 1.2) if not pd.isna(max_epoch) else 2000
    )
    
    # Pivot the table
    pivot_table = df_plot.pivot_table(
        values='convergence_epoch', 
        index=param1, 
        columns=param2, 
        aggfunc='mean'
    )
    
    # Plot
    sns.heatmap(pivot_table, annot=True, cmap='viridis_r', fmt=".0f")
    plt.title(f'Epochs to Convergence: {param1} vs {param2}')
    plt.tight_layout()
    plt.show()

def plot_predictions(rnn,batch_size =4,is_reversed=False, model_type="sl", run_dir=None):
    
    with torch.no_grad(): #Do not track gradients 
        if model_type == "sl":
        #Make fresh batch for testing
            inputs, targets, lengths, mask, cues = generate_batch(
                trial_params=TASK,
                is_reversed=is_reversed,
                cues=["A","B","B","C"],
                rewards=[1,1,0,0] if not is_reversed else [0,0,1,1])

            ys, hs = rnn.forward(inputs)

            #Convert to numpy for plotting 
            ys_np = ys.detach().numpy()   
            ys_np = ys.detach().numpy()
            plot_inputs = inputs.detach().numpy()
        else:
            env = RLTask(batch_size=batch_size)
            
            stimuli_idx = [0, 1, 1, 2][:batch_size]
            cues = [['A', 'B', 'C'][s] for s in stimuli_idx]
            lengths = torch.tensor(env.seq_lens[:batch_size])
            
            inputs = torch.zeros((batch_size, env.max_seq_len, 3))

            # Populate the inputs by iterating over the batch dimension
            for b in range(batch_size):
                stim_class = stimuli_idx[b]
                start = env.stim_starts[b]
                end = env.stim_ends[b]
                # Set the one-hot encoding only during the specific stimulus window for this trial
                inputs[b, start:end, stim_class] = 1.0

            action_probs, values = rnn.rl_model(inputs)
            _, hs = rnn.forward(inputs)

        # Append a dummy 4th channel (zeros) so imshow() draws the exact same 4-row grid as sl
            dummy_channel = torch.zeros((batch_size, env.max_seq_len, 1))
            plot_inputs = torch.cat([inputs, dummy_channel], dim=-1).detach().numpy()


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
        


        x_axis = np.arange(trial_len) #ensure lines stop when trial ends

        if model_type == "sl":
            pred_seq = ys[i, :trial_len, 0].detach().numpy()
            y_pred = 3 - (pred_seq * 3)
            target_seq = targets[i, :trial_len, 0].detach().numpy()        #Overlay the lines for target and moel prediction
            y_target = 3 - (target_seq * 3)

            axes[i].plot(x_axis, y_target, color='red', label='SL Target', linewidth=2)
            axes[i].plot(x_axis, y_pred, color='green', label='RNN Prediction', linestyle='--')

            #Labels and formatting
            axes[i].set_yticks([0, 1, 2, 3])
            axes[i].set_yticklabels(['A', 'B', 'C', 'R'])
        else:
            critic_seq = values[i, :trial_len].squeeze().detach().numpy()
            
            # Plot the raw Expected Value (no need to multiply by 3)
            axes[i].plot(x_axis, critic_seq, color='purple', label='Critic Expected Value', linewidth=2)
            
            # Format the Y-axis to show real EV numbers instead of percentages
            axes[i].set_yticks([0.0, 0.5, 1.0])
            axes[i].set_yticklabels(['0.0 (EV)', '0.5 (EV)', '1.0 (EV)'])
            axes[i].set_ylim([-0.1, 1.1]) # Give the line a little breathing room
        axes[i].set_ylabel(f"Trial {i}\n({cues[i]})")

    axes[-1].set_xlabel("Timestep")
    # Only show legend on the first subplot
    axes[0].legend(loc='upper right', fontsize='small')
    plt.suptitle(f"Model Predictions ({'Reversed' if is_reversed else 'Baseline'} - {model_type.upper()})", y=1.02)
    plt.tight_layout()
    if run_dir:
        suffix = "reversed" if is_reversed else "normal"
        plt.savefig(Path(run_dir) / f"predictions_{model_type}_{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pca_mean_trajectories(
        hs,
        cues, 
        lengths, 
        inputs, 
        is_reversed=False, 
        pca=None, 
        run_dir=None, 
        custom_title=None, 
        events=None):
    
    from matplotlib.lines import Line2D
    import scipy.stats as stats
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    hs_np = hs.detach().numpy()
    inputs_np = inputs.detach().numpy()
    batch_size = hs_np.shape[0]

    # Fit PCA on the entire flattened trajectory history instead of a snapshot.
    passed_pca = False
    if pca is None:
        flat_valid_hs = []
        for i in range(batch_size):
            trial_len = lengths[i].item()
            flat_valid_hs.append(hs_np[i, :trial_len, :])
        
        flat_valid_hs = np.concatenate(flat_valid_hs, axis=0)
        pca = PCA(n_components=2).fit(flat_valid_hs)
    else:
        print("Passed a PCA -> Cross-Projecting into provided space")
        passed_pca = True
    
    # --- SPLIT BIMODAL OUTCOMES (50% REWARD) ---
    conditions = ['A', 'B_rew', 'B_unrew', 'C']
    aligned_pca = {k: [] for k in conditions}
    aligned_inp = {k: [] for k in conditions}
    aligned_events = {
        k: {'plot_start': [], 'stim_on': [], 'stim_last': [], 'reward_on': [], 'reward_last': [], 'trial_end': []}
        for k in conditions
    }

    for i in range(batch_size):
        trial_len = lengths[i].item()
        cue = cues[i]
        
        trial_pca = pca.transform(hs_np[i, :trial_len, :])
        trial_in = inputs_np[i, :trial_len, :] # Keep all channels to check reward
        
        # Determine if trial was rewarded (Checks channel 3 in Supervised Learning)
        is_rewarded = False
        if trial_in.shape[1] >= 4: 
            is_rewarded = np.any(trial_in[:, 3] > 0.5)
        
        # Split Stimulus B based on outcome
        if cue == 'B':
            cue_key = 'B_rew' if is_rewarded else 'B_unrew'
        else:
            cue_key = cue
            
        if events is not None:
            t_stim_start = int(events[i]["stim_on"])
            t_stim_last = int(events[i]["stim_last"])
            t_reward_on = int(events[i]["reward_on"])
            t_reward_last = int(events[i]["reward_last"])
            t_trial_end = int(events[i]["trial_end"])
        else:
            stim_activity = np.sum(trial_in[:, 0:3], axis=1) 
            cue_active = np.where(stim_activity > 0.5)[0]

            if len(cue_active) == 0:
                continue

            t_stim_start = int(cue_active[0])
            t_stim_last = int(cue_active[-1])
            t_reward_on = min(t_stim_last + 1, trial_len - 1) 
            t_reward_last = trial_len - 1
            t_trial_end = trial_len - 1

        start_idx = max(0, t_stim_start - 2)
        traj_pca = trial_pca[start_idx : trial_len, :]
        traj_in = trial_in[start_idx : trial_len, 0:3] 
                
        # Pad the beginning if needed
        pad_len = max(0, (t_stim_start - 2) * -1)
        if pad_len > 0:
            traj_pca = np.vstack([np.tile(traj_pca[0, :], (pad_len, 1)), traj_pca])
            traj_in = np.vstack([np.zeros((pad_len, 3)), traj_in])
            
        aligned_pca[cue_key].append(traj_pca)
        aligned_inp[cue_key].append(traj_in)

        aligned_events[cue_key]['plot_start'].append(0)
        aligned_events[cue_key]['stim_on'].append(t_stim_start - start_idx + pad_len)
        aligned_events[cue_key]['stim_last'].append(t_stim_last - start_idx + pad_len)
        aligned_events[cue_key]['reward_on'].append(t_reward_on - start_idx + pad_len)
        aligned_events[cue_key]['reward_last'].append(t_reward_last - start_idx + pad_len)
        aligned_events[cue_key]['trial_end'].append(t_trial_end - start_idx + pad_len)
    
    fig, ax = plt.subplots(figsize=(11, 8))
    
    colors = {
        'A': '#086ec7', 
        'B_rew': '#f5a700', 
        'B_unrew': '#b37a00', # Darker orange for unrewarded
        'C': '#31A354'
    }
    
    if is_reversed:
        labels = {
            'A': 'Stim A (0%)', 
            'B_rew': 'Stim B (50%) - Rewarded', 
            'B_unrew': 'Stim B (50%) - No Reward', 
            'C': 'Stim C (100%)'
        }
    else:
        labels = {
            'A': 'Stim A (100%)', 
            'B_rew': 'Stim B (50%) - Rewarded', 
            'B_unrew': 'Stim B (50%) - No Reward', 
            'C': 'Stim C (0%)'
        }
    
    for cue_key in conditions:
        if not aligned_pca[cue_key]: continue
        
        min_len = min(len(t) for t in aligned_pca[cue_key])
        stacked_pca = np.array([t[:min_len] for t in aligned_pca[cue_key]])
       
        mean_pca = np.mean(stacked_pca, axis=0)
        se_pca = np.std(stacked_pca, axis=0)
        
        x_m, y_m = mean_pca[:, 0], mean_pca[:, 1]
        x_se, y_se = se_pca[:, 0], se_pca[:, 1]
        c = colors[cue_key]

        dx = np.gradient(x_m)
        dy = np.gradient(y_m)

        norm = np.sqrt(dx**2 + dy**2) + 1e-8
        nx = -dy / norm
        ny = dx / norm

        radial_sd = np.sqrt(x_se**2 + y_se**2)

        upper = np.column_stack([
            x_m + nx * radial_sd,
            y_m + ny * radial_sd
        ])

        lower = np.column_stack([
            x_m - nx * radial_sd,
            y_m - ny * radial_sd
        ])

        poly = np.vstack([upper, lower[::-1]])

        ax.fill(
            poly[:, 0],
            poly[:, 1],
            color=c,
            alpha=0.2,
            edgecolor="none",
            zorder=2
        )
        
        # Make the unrewarded line dashed for clarity
        line_style = '--' if 'unrew' in cue_key else '-'
        ax.plot(x_m, y_m, color=c, linewidth=2.5, linestyle=line_style, label=labels[cue_key], zorder=4)
        
        def median_event_idx(event_name):
            if not aligned_events[cue_key][event_name]:
                return 0
            idx = int(np.round(np.median(aligned_events[cue_key][event_name])))
            return int(np.clip(idx, 0, min_len - 1))

        idx_start = median_event_idx("plot_start")
        idx_stim_on = median_event_idx("stim_on")
        idx_stim_last = median_event_idx("stim_last")
        idx_reward_on = median_event_idx("reward_on")
        idx_trial_end = median_event_idx("trial_end")

        ax.scatter(x_m[idx_start], y_m[idx_start], color='black', marker='o', s=40, zorder=6)
        ax.scatter(x_m[idx_stim_on], y_m[idx_stim_on], color='black', marker='^', s=80, zorder=6)
        ax.scatter(x_m[idx_stim_last], y_m[idx_stim_last], color='black', marker='d', s=80, zorder=7)
        ax.scatter(x_m[idx_reward_on], y_m[idx_reward_on], color='black', marker='s', s=75, zorder=8)
        ax.scatter(x_m[idx_trial_end], y_m[idx_trial_end], color=c, marker='x', s=120, zorder=9, linewidths=3)
        
    # --- Formatting & Dual Legends ---
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    plt.title(custom_title if custom_title else "Unified PCA Trajectories", fontsize=14, pad=15)
    plt.xlabel(f"Principal Component 1 ({pc1_var:.1f}% variance)")
    plt.ylabel(f"Principal Component 2 ({pc2_var:.1f}% variance)")
    
    handles, legend_labels = ax.get_legend_handles_labels()
    stim_legend = ax.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(0.0, -0.12), 
                            title="Stimulus Identity & Outcome", frameon=True)
    ax.add_artist(stim_legend) 
    
    phase_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Trial Start'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label='Stimulus Onset'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='black', markersize=10, label='Stimulus Offset'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=9, label='Reward Onset'),
        Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, markeredgewidth=2, label='Trial End')
    ]
    ax.legend(handles=phase_elements, loc='upper right', bbox_to_anchor=(1.0, -0.12), 
              title="Phase Key", ncol=2, frameon=True)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.subplots_adjust(bottom=0.25) 
    
    if run_dir:
        if passed_pca: 
            suffix = "cross_pca"
        elif is_reversed:
            suffix = "reversed"
        else:
            suffix = "normal"
        plt.savefig(Path(run_dir) / f"pca_trajectory_{suffix}.png", dpi=300)
        plt.close()
    else:
        plt.show()
def animate_pca_trajectories(weight_paths, model_type="sl",is_reversed=False, num_trials_per_stim=100, run_dir=None):
    # 1. Initialize network and generate fixed input data
    # Use the exact same trials for every epoch to cleanly see learning
    if model_type == "rl":
        rnn = RLModelWrapper(ActorCriticRNN(input_size=3, hidden_size=64, num_actions=2))
    else:
        rnn = ScratchRNN()
    
    
    cues = ["A"] * num_trials_per_stim + ["B"] * num_trials_per_stim + ["C"] * num_trials_per_stim
    
    with torch.no_grad():
        inputs, targets, lengths, mask, cues = generate_batch(
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
    all_x = []
    all_y = []

    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    for i, epoch_hs in enumerate(all_epochs_hs):
        epoch_pca = []
        for i in range(batch_size):
            trial_len = lengths[i].item()
            trial_hs = epoch_hs[i, :trial_len, :]
            transformed = pca.transform(trial_hs)
            epoch_pca.append(transformed)
            
            # Update global bounds so the camera stays still
            all_x.extend(transformed[:, 0])
            all_y.extend(transformed[:, 1])
            
        all_epochs_pca.append(epoch_pca)

    x_min, x_max = np.percentile(all_x, [2, 98])
    y_min, y_max = np.percentile(all_y, [2, 98])
        
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
        print(f"Working on frame: {frame_idx}")
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

def plot_clusters(hs, cues, lengths, is_reversed=False, run_dir=None):
    """Fits PCA to a single snapshot in time (end of delay) to show state clusters."""
    # Convert to numpy
    hs_np = hs.detach().numpy()
    batch_size = hs_np.shape[0]
    
    #Isolate the specific timestep (Snapshot)
    snapshot_hs = []
    for i in range(batch_size):
        trial_len = lengths[i].item()
        
        #get the timestep right BEFORE the reward is delivered.
        anticipation_timestep = trial_len - 6
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
    if run_dir:
        suffix = "reversed" if is_reversed else "normal"
        plt.savefig(Path(run_dir) / f"pca_clusters_{suffix}.png", dpi=300)
        plt.close() # Prevents memory leaks by closing the figure instead of showing it
    else:
        plt.show()

def plot_distance_matrix(hs, cues, lengths, is_reversed=False, run_dir=None):
    #Calculate the euclidean distance between representations of stimuli
    hs_np = hs.detach().numpy()
    batch_size = hs_np.shape[0]

    snapshots = {'A': [], 'B': [], 'C': []} 
    #Get the same snapshots as in the cluster plot
    for i in range(batch_size):
        trial_len = lengths[i].item()
        anticipation_timestep = trial_len - 6 #get the same timestep just before the reward

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
    if run_dir:
        suffix = "reversed" if is_reversed else "normal"
        plt.savefig(Path(run_dir) / f"pca_distance_matrix_{suffix}.png", dpi=300)
        plt.close() # Prevents memory leaks by closing the figure instead of showing it
    else:
        plt.show()  

def plot_batch_timeline(live_performance, batches_per_epoch, reversal_epoch, run_dir=None, reversed=False):
    total_batches = len(live_performance['A'])
    x_axis = np.arange(total_batches)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the raw high-frequency batch data with some transparency
    plt.plot(x_axis, live_performance['A'], color='blue', alpha=0.8, linewidth=1.5, label='Stimulus A')
    plt.plot(x_axis, live_performance['B'], color='orange', alpha=0.8, linewidth=1.5, label='Stimulus B (50%)')
    plt.plot(x_axis, live_performance['C'], color='green', alpha=0.8, linewidth=1.5, label='Stimulus C')
    
    # Mark Reversal Point (convert epoch to batch index)
    reversal_batch = reversal_epoch * batches_per_epoch
    plt.axvline(x=reversal_batch, color='black', linestyle='--', linewidth=2.5, label='Reversal Initiated')
    
    # Target Guidelines
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.6)
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6)
    plt.axhline(y=0.0, color='gray', linestyle=':', alpha=0.6)
    
    # Create secondary X-axis for Epochs
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    
    # Set epoch ticks (e.g., every 50 epochs)
    epoch_ticks = np.arange(0, (total_batches // batches_per_epoch) + 1, 30)
    batch_ticks = epoch_ticks * batches_per_epoch
    ax1.set_xticks(batch_ticks)
    ax2.set_xticks(batch_ticks)
    ax2.set_xticklabels(epoch_ticks)
    
    ax1.set_xlabel("Training Batch")
    ax2.set_xlabel("Training Epoch")
    ax1.set_ylabel("Predicted Value")#Lick Rate")
    plt.title("High-Resolution Batch Evolution of Anticipatory Predictions", pad=20)
    ax1.legend(loc='center right')
    
    plt.tight_layout()
    if run_dir:
        suffix = "reversed" if reversed else "normal"
        plt.savefig(Path(run_dir) / f"batch_timeline_{suffix}.png", dpi=300)
        plt.close() # Prevents memory leaks by closing the figure instead of showing it
    else:
        plt.show() 

def plot_decoding_stimulus(accuracies_dict, avg_accuracy, avg_stimulus_start_timestep, avg_stimulus_end_timestep, reversed=False,run_dir=None):
    #Plots the decoding of stimulus identity
    plt.figure(figsize=(8, 5))
    
    # Assign distinct colors 
    # Make the same ----
    colors = {'A': 'crimson', 'B': 'forestgreen', 'C': 'royalblue'}
    
    # Plot each cue's accuracy trajectory
    for cue, acc_list in accuracies_dict.items():
        times = np.arange(len(acc_list))
        color = colors.get(cue, 'black') # Fallback to black if cue isn't in dict
        
        plt.plot(times, acc_list, marker='o', markersize=4, linewidth=2, 
                 label=f'Stimulus {cue}', color=color)
    
    if avg_stimulus_start_timestep is not None and avg_stimulus_end_timestep is not None:
        plt.axvline(x=avg_stimulus_start_timestep, color='black', linestyle=':', linewidth=2)
        plt.axvline(x=avg_stimulus_end_timestep, color='black', linestyle=':', linewidth=2)
        
        midpoint = (avg_stimulus_start_timestep + avg_stimulus_end_timestep) / 2
        plt.text(
            midpoint,
            -0.08,  # slightly below x-axis
            "Stimulus Window",
            ha='center',
            va='top',
            transform=plt.gca().get_xaxis_transform()
        )

    times = np.arange(len(avg_accuracy))
        
    plt.plot(times, avg_accuracy,alpha=0.4, marker='o', markersize=2, linewidth=2, 
            label='Avg accuracy', color="black", linestyle='--')


    # Plot chance level
    # If ever introduce reversal trials with more/fewer cues, change 1/3 accordingly
    plt.axhline(y=1/3, color='gray', linestyle='--', label='Chance (33%)')


    # Formatting
    plt.title("Decoding Accuracy by Stimulus Identity")
    plt.xlabel("Time Step")
    plt.ylabel("Accuracy (True Positive Rate)")
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if run_dir:
        suffix = "reversed" if reversed else "normal"
        plt.savefig(Path(run_dir) / f"decoding_stimulus_{suffix}.png", dpi=300)
        plt.close() # Prevents memory leaks by closing the figure instead of showing it
    else:
        plt.show()

def plot_decoding_trajectories(mean_predictions, avg_stimulus_start_timestep=None, avg_stimulus_end_timestep=None, reversed=False, run_dir=None):
    #Plots decoding of expected lick rate by cue type.
    plt.figure(figsize=(8, 5))
    
    # Match the colors used in the classifier plot
    colors = {'A': 'crimson', 'B': 'forestgreen', 'C': 'royalblue'}
    
    for cue, trajectory in mean_predictions.items():
        times = np.arange(len(trajectory))
        color = colors.get(cue, 'black')
        
        plt.plot(times, trajectory, linewidth=2, label=f'Cue {cue}', color=color)

    # --- Add Stimulus Boundaries ---
    if avg_stimulus_start_timestep is not None and avg_stimulus_end_timestep is not None:
        plt.axvline(x=avg_stimulus_start_timestep, color='black', linestyle=':', linewidth=2)
        plt.axvline(x=avg_stimulus_end_timestep, color='black', linestyle=':', linewidth=2)
        
        midpoint = (avg_stimulus_start_timestep + avg_stimulus_end_timestep) / 2
        plt.text(
            midpoint,
            -0.08,  # slightly below x-axis
            "Stimulus Window",
            ha='center',
            va='top',
            transform=plt.gca().get_xaxis_transform()
        )
    # -------------------------------

    plt.title("Decoded Behavioral Trajectories (Predicted Lick Rate)")
    plt.xlabel("Time Step")
    plt.ylabel("Predicted Target Value")
    
    # Upper left so it doesn't overlap the climbing trajectories
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if run_dir:
        suffix = "reversed" if reversed else "normal"
        plt.savefig(Path(run_dir) / f"decoding_trajectory_{suffix}.png", dpi=300)
        plt.close() # Prevents memory leaks by closing the figure instead of showing it
    else:
        plt.show()
# ---- UTILITIES ------
def initialize_run_directory():

    # Generate the Run ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_hash = uuid.uuid4().hex[:5]
    run_id = f"run_{timestamp}_{unique_hash}"
    
    # Define the path
    run_dir = Path(f"./results/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Pull settings from existing TRAINING dictionary and save them
    # Ensures  random seed and hyperparameters are added to the run_id
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        **TRAINING # Unpacks all hyperparameters from configs.py TRAINING 
    }
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f" Run Directory Initialized: {run_dir}")
    return run_dir

def load_live_performance():
    try:
        live_performance = torch.load("checkpoints/live_performance.pt", weights_only=False)
        print("Loaded live performance data")
        return live_performance
    except FileNotFoundError:
        raise FileNotFoundError("No saved data found. Need to run train_model() first")

def extract_epoch(filename):
    match = re.search(r'weights_epoch_(\d+)', filename)
    return int(match.group(1)) if match else -1

def get_checkpoint_files(reversal_epoch):
    files = glob.glob("checkpoints/weights_epoch_*.pth")
    files.sort(key=extract_epoch)

    if Path("checkpoints/weights_baseline.pth").exists():
        files.append("checkpoints/weights_baseline.pth")

    return files

# ---- HELPERS TO RUN PLOTS CLEANLY ----
def run_timeline_plot(live_performance, cfg,run_dir=None):
    plot_batch_timeline(live_performance, cfg.batches_per_epoch, cfg.reversal_epoch, run_dir=run_dir)

def run_acquisition_animation(checkpoint_files, cfg, model_type="sl"):
    acquisition_files = [f for f in checkpoint_files if extract_epoch(f) < cfg.reversal_epoch]
    acquisition_files.append("checkpoints/weights_baseline.pth")

    print(f"Animating Acquisition: {len(acquisition_files)} frames...")
    anim = animate_pca_trajectories(
        acquisition_files,
        model_type=model_type,
        is_reversed=False,
        num_trials_per_stim=cfg.num_trials_per_stim,
    )
    anim.save("pca_acquisition.mp4", writer="ffmpeg", fps=cfg.fps)

def run_reversal_animation(checkpoint_files, cfg, model_type="sl"):
    reversal_files = ["checkpoints/weights_baseline.pth"] + [
        f for f in checkpoint_files if extract_epoch(f) >= cfg.reversal_epoch
    ]

    print(f"Animating Reversal: {len(reversal_files)} frames...")
    anim = animate_pca_trajectories(
        reversal_files,
        model_type=model_type,
        is_reversed=True,
        num_trials_per_stim=cfg.num_trials_per_stim,
    )
    anim.save("pca_reversal.mp4", writer="ffmpeg", fps=cfg.fps)

def run_baseline_plots(cfg, model_type="sl", run_dir=None):
    if model_type == "rl":
        rnn = RLModelWrapper(ActorCriticRNN(input_size=3, hidden_size=64, num_actions=2))
    else:
        rnn = ScratchRNN()
    rnn.load_state_dict(torch.load("checkpoints/weights_baseline.pth"))

    if model_type == "rl":
        hs, cues, lengths, inputs, events = extract_hidden_states_rl(
            model=rnn,
            num_trials_per_stim=cfg.num_trials_per_stim
        )
    else:
        hs, cues, lengths, inputs, targets = extract_hidden_states(
            model=rnn,
            trial_params=TASK,
            num_trials_per_stim=cfg.num_trials_per_stim,
            is_reversed=False
        )
        events = None

    print("plotting baseline trajectories ")
    plot_pca_mean_trajectories(
        hs=hs, 
        cues=cues, 
        lengths=lengths, 
        inputs=inputs, 
        is_reversed=False, 
        run_dir=run_dir,
        custom_title=f"Standard Mean PCA Trajectories (Pre-Reversal)",
        events=events
    )
    
    plot_clusters(hs, cues, lengths, is_reversed=False, run_dir=run_dir)
    plot_distance_matrix(hs, cues, lengths, is_reversed=False, run_dir=run_dir)
    plot_predictions(rnn, is_reversed=False, model_type=model_type, run_dir=run_dir)

def run_final_reversal_plots(cfg, model_type="sl", run_dir=None):
    if model_type == "rl":
        rnn = RLModelWrapper(ActorCriticRNN(input_size=3, hidden_size=64, num_actions=2))
    else:
        rnn = ScratchRNN()
    rnn.load_state_dict(torch.load("checkpoints/weights_final_reversal.pth"))

    hs, cues, lengths, inputs, targets = extract_hidden_states(
        model=rnn, 
        trial_params=TASK,  
        num_trials_per_stim=cfg.num_trials_per_stim, 
        is_reversed=True
    )
    if model_type == "rl":
        hs, cues, lengths, inputs, events = extract_hidden_states_rl(
            model=rnn,
            num_trials_per_stim=cfg.num_trials_per_stim
        )
    else:
        hs, cues, lengths, inputs, targets = extract_hidden_states(
            model=rnn,
            trial_params=TASK,
            num_trials_per_stim=cfg.num_trials_per_stim,
            is_reversed=False
        )
        events = None
    print("plotting final reversal trajectories ")
    plot_pca_mean_trajectories(
        hs=hs, 
        cues=cues, 
        lengths=lengths, 
        inputs=inputs,  
        is_reversed=True,
        run_dir=run_dir,
        custom_title=f"Standard Mean PCA Trajectories (Post-Reversal)",
        events=events
    )
    plot_clusters(hs, cues, lengths, is_reversed=True, run_dir=run_dir)
    plot_distance_matrix(hs, cues, lengths, is_reversed=True, run_dir=run_dir)
    
    plot_predictions(rnn, is_reversed=True, model_type=model_type, run_dir=run_dir)

def run_decoding_analysis(model, trial_params,run_dir, baseline_ckpt="checkpoints/weights_baseline.pth", reversal_ckpt="checkpoints/weights_final_reversal.pth"):
    #Pre-reversal
    print("--- Starting Pre-Reversal Decoding Analysis ---")
    model.load_state_dict(torch.load(baseline_ckpt))

    accuracies_dict, avg_accuracy, avg_stimulus_start_timestep, avg_stimulus_end_timestep = train_stimulus_decoders(
        model=model, 
        trial_params=trial_params, 
        reversed=False
    )
    plot_decoding_stimulus(
        accuracies_dict, avg_accuracy, avg_stimulus_start_timestep, avg_stimulus_end_timestep, 
        reversed=False, run_dir=run_dir
    )

    # Pre-reversal Continuous Decoding
    mean_predictions = train_continuous_decoders(
        model=model, 
        trial_params=trial_params, 
        reversed=False
    )
    plot_decoding_trajectories(
        mean_predictions, 
        avg_stimulus_start_timestep=avg_stimulus_start_timestep, 
        avg_stimulus_end_timestep=avg_stimulus_end_timestep, 
        reversed=False, run_dir=run_dir
    )

    print("Plotting trajectories with fake SE (single run)...")
    
    print("--- Starting Post-Reversal Decoding Analysis ---")
    model.load_state_dict(torch.load(reversal_ckpt))
    accuracies_dict, avg_accuracy, avg_stimulus_start_timestep, avg_stimulus_end_timestep = train_stimulus_decoders(
        model=model, 
        trial_params=trial_params, 
        reversed=True
    )
    plot_decoding_stimulus(
        accuracies_dict, avg_accuracy, avg_stimulus_start_timestep, avg_stimulus_end_timestep, 
        reversed=True, run_dir=run_dir
    )

    # Post-reversal Continuous Decoding
    mean_predictions = train_continuous_decoders(
        model=model, 
        trial_params=trial_params, 
        reversed=True
    )
    plot_decoding_trajectories(
        mean_predictions, 
        avg_stimulus_start_timestep=avg_stimulus_start_timestep, 
        avg_stimulus_end_timestep=avg_stimulus_end_timestep, 
        reversed=True, run_dir=run_dir
    )
    
    print("Decoding Analysis Complete.")

def run_cross_projection_pca(
        model, 
        trial_params, 
        baseline_ckpt, 
        reversal_ckpt, 
        run_dir=None,
        model_type="sl"):

    # Fits PCA on baseline (pre-reversal) hidden states, and plots 
    # the post-reversal hidden states projected into that exact same space.

    print("--- Running Cross-Condition PCA Projection ---")
    
    # Extract Baseline & Fit the "Clean" PCA Space
    model.load_state_dict(torch.load(baseline_ckpt))

    if model_type == "rl":
        hs_base, cues_base, lengths_base, inputs_base, events_base = extract_hidden_states_rl(
            model=model,
            num_trials_per_stim=100
        )
    else:
        hs_base, cues_base, lengths_base, inputs_base, _ = extract_hidden_states(
            model=model,
            trial_params=trial_params,
            is_reversed=False
        )
        events_base = None
        
    # Convert baseline hidden states to a flat numpy array to fit PCA
    hs_base_np = hs_base.detach().numpy()

    # --- Fit PCA on the entire flattened baseline trajectory history ---
    flat_valid_hs_base = []
    for i in range(len(cues_base)):
        trial_len = lengths_base[i].item()
        flat_valid_hs_base.append(hs_base_np[i, :trial_len, :])
        
    flat_valid_hs_base = np.concatenate(flat_valid_hs_base, axis=0)
    
    # Fit the PCA on the dynamic baseline geometry
    fixed_pca = PCA(n_components=2)
    fixed_pca.fit(flat_valid_hs_base) 
    
    # Extract Reversal Data & Plot in Baseline Space
    model.load_state_dict(torch.load(reversal_ckpt))

    if model_type == "rl":
        hs_rev, cues_rev, lengths_rev, inputs_rev, events_rev = extract_hidden_states_rl(
            model=model,
            num_trials_per_stim=100
        )
    else:
        hs_rev, cues_rev, lengths_rev, inputs_rev, _ = extract_hidden_states(
            model=model,
            trial_params=trial_params,
            is_reversed=True
        )
        events_rev = None
    
    print("Projecting Reversal trajectories into Baseline PCA axes...")
    
    plot_pca_mean_trajectories(
        hs=hs_rev,
        cues=cues_rev,
        lengths=lengths_rev,
        inputs=inputs_rev,
        is_reversed=True,
        pca=fixed_pca,
        run_dir=run_dir,
        custom_title="Post-Reversal Mean Trajectories (Projected in Baseline PCA Space)",
        events=events_rev,
    )

def aggregate_continuous_decoding(model, trial_params, checkpoints, reversed=False):

    #Runs continuous decoding across multiple model checkpoints and calculates mean and SE.

    all_runs_predictions = {'A': [], 'B': [], 'C': []}
    
    for ckpt in checkpoints:
        # Load the specific seed's weights
        model.load_state_dict(torch.load(ckpt))
        
        # Get predictions for this run
        predictions = train_continuous_decoders(model, trial_params, reversed=reversed)
        
        for cue in ['A', 'B', 'C']:
            all_runs_predictions[cue].append(predictions[cue])
            
    # Calculate Mean and Standard Error
    summary_stats = {}
    for cue in ['A', 'B', 'C']:
        stacked = np.vstack(all_runs_predictions[cue]) # Shape: (num_runs, sequence_length)
        mean_traj = np.mean(stacked, axis=0)
        # Standard Error = Standard Deviation / sqrt(N)
        se_traj = stats.sem(stacked, axis=0) 
        summary_stats[cue] = {'mean': mean_traj, 'se': se_traj}
        
    return summary_stats

def plot_trajectories_with_error(summary_stats, avg_stimulus_start=None, avg_stimulus_end=None, reversed=False, run_dir=None):
    plt.figure(figsize=(8, 5))
    colors = {'A': 'crimson', 'B': 'forestgreen', 'C': 'royalblue'}
    
    for cue, stats in summary_stats.items():
        mean = stats['mean']
        se = stats['se']
        times = np.arange(len(mean))
        color = colors.get(cue, 'black')
        
        # Plot the main trajectory line
        plt.plot(times, mean, linewidth=2, label=f'Cue {cue}', color=color)
        # Fill the standard error region
        plt.fill_between(times, mean - se, mean + se, color=color, alpha=0.2)

    # --- Add Stimulus Boundaries ---
    if avg_stimulus_start is not None and avg_stimulus_end is not None:
        plt.axvline(x=avg_stimulus_start, color='black', linestyle=':', linewidth=2)
        plt.axvline(x=avg_stimulus_end, color='black', linestyle=':', linewidth=2)
        midpoint = (avg_stimulus_start + avg_stimulus_end) / 2
        plt.text(midpoint, -0.08, "Stimulus Window", ha='center', va='top', transform=plt.gca().get_xaxis_transform())

    plt.title("Decoded Trajectories across Multiple Initializations (Mean ± SE)")
    plt.xlabel("Time Step")
    plt.ylabel("Predicted Target Value")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if run_dir:
        suffix = "reversed" if reversed else "normal"
        plt.savefig(Path(run_dir) / f"decoding_trajectory_SE_{suffix}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_all_graphs(train=False, model_type="sl", plots=None, run_dir=None):
    cfg = PlotConfig(
        lr=TRAINING["lr"],
        batch_size=TRAINING["batch_size"],
        epochs=TRAINING["epochs"],
        batches_per_epoch=TRAINING["batches_per_epoch"],
        reversal_epoch=TRAINING["reversal_epoch"],
    )

    if plots is None:
        plots = ["timeline", "acq_anim", "rev_anim", "baseline", "final_reversal"]

    plots = set(plots)

    if train:
        if model_type == "rl":
            # Direct routing to your RL loop
            _, _, live_performance = train_rl_model(
                num_epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr
            )
        else:
            my_trial_counts = {"A": 10, "B": 10, "C": 12}
            dataset = generate_full_dataset(
                epochs=cfg.epochs,
                trial_params=TASK,             
                trial_counts=my_trial_counts, 
                batches_per_epoch=cfg.batches_per_epoch,
                reversal_epoch=cfg.reversal_epoch,
            )
            _, _, live_performance = train_model(
                ScratchRNN(), dataset, cfg.lr, cfg.epochs, cfg.batches_per_epoch, cfg.batch_size, probe=True
            )
    else:
        live_performance = load_live_performance()

    checkpoint_files = get_checkpoint_files(cfg.reversal_epoch)

    if "timeline" in plots:
        run_timeline_plot(live_performance, cfg, run_dir=run_dir)
    if "acq_anim" in plots:
        run_acquisition_animation(checkpoint_files, cfg, model_type=model_type)
    if "rev_anim" in plots:
        run_reversal_animation(checkpoint_files, cfg, model_type=model_type)
    if "baseline" in plots:
        run_baseline_plots(cfg, model_type=model_type, run_dir=run_dir)
    if "final_reversal" in plots:
        run_final_reversal_plots(cfg, model_type=model_type, run_dir=run_dir)
    if "cross_pca" in plots:
        run_cross_projection_pca(
            model=ScratchRNN() if model_type == "sl" else RLModelWrapper(ActorCriticRNN(input_size=3, hidden_size=64, num_actions=2)),
            trial_params=TASK,
            baseline_ckpt="checkpoints/weights_baseline.pth",
            reversal_ckpt="checkpoints/weights_final_reversal.pth",
            run_dir=run_dir
        )
    if "heatmap" in plots:
        if model_type == "sl":
            df_results= run_hyperparameter_search()
            plot_heatmap(df_results, param1='hidden_size', param2='target_sigma')
        elif model_type == "rl":
            df_results= run_rl_hyperparameter_search()
            plot_rl_convergence_heatmap(df_results, param1='gamma', param2='entropy_coef')
    if "decoding" in plots:
        
        if model_type == "sl":
            # If analyzing the Supervised model:
            my_model = ScratchRNN(input_size=4, hidden_size=256, output_size=1)

            run_decoding_analysis(model=my_model, trial_params=TASK, run_dir=run_dir,
                                  baseline_ckpt="checkpoints/weights_baseline.pth",
                                  reversal_ckpt="checkpoints/weights_final_reversal.pth")
        # If analyzing the Reinforcement Learning model:
        else:
            base_rl = ActorCriticRNN(input_size=3, hidden_size=64)
            my_wrapped_model = RLModelWrapper(base_rl)
            run_decoding_analysis(model=my_wrapped_model, trial_params=TASK, run_dir=run_dir)

if __name__ == "__main__":
    # 1. Ask user which framework model to isolate
    model_choice = input("Choose model framework to plot (sl / rl): ").strip().lower()
    if model_choice not in ["sl", "rl"]:
        model_choice = "sl" # Default fallback
        
    save = input("Save graphs? (y/n): ").lower()
    current_run_dir = initialize_run_directory() if save == "y" else None
    
    # Run the setup cleanly using the choice string
    plot_all_graphs(train=True, model_type=model_choice, plots=["baseline", "final_reversal","cross_pca"], run_dir=current_run_dir)
