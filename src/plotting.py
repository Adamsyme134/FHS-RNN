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
from scipy.ndimage import gaussian_filter1d
from configs import *

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

def plot_rl_convergence_heatmap(df_results, param1='bptt_horizon', param2='gamma'):
    plt.figure(figsize=(8, 6))
    df_plot = df_results.copy()
    
    # 1. Figure out a penalty color value for the failed runs
    valid_runs = df_plot[df_plot['convergence_epoch'] != "Did Not Converge"]
    max_epoch = valid_runs['convergence_epoch'].max() if not valid_runs.empty else 2000
    penalty_val = int(max_epoch * 1.2)
    
    # 2. Create a purely numerical column for Seaborn's color mapping
    df_plot['color_val'] = df_plot['convergence_epoch'].apply(
        lambda x: penalty_val if x == "Did Not Converge" else x
    )
    
    # 3. Pivot the numeric data for the colors (This preserves axis labels!)
    pivot_color = df_plot.pivot_table(
        values='color_val', index=param1, columns=param2, aggfunc='mean'
    )
    
    # 4. Pivot the raw data for the text annotations and replace failures with "N/A"
    pivot_annot = df_plot.pivot_table(
        values='convergence_epoch', index=param1, columns=param2, aggfunc=lambda x: x.iloc[0]
    )
    annot_matrix = pivot_annot.replace("Did Not Converge", "N/A").values
    
    # 5. Plot! (Pass the custom string matrix to 'annot' and format as strings using fmt="")
    sns.heatmap(pivot_color, annot=annot_matrix, fmt="", cmap='viridis_r', 
                cbar_kws={'label': 'Epochs to Converge'})
    
    plt.title(f'Epochs to Convergence: {param1} vs {param2}')
    plt.tight_layout()
    plt.show()

def plot_predictions(rnn, batch_size=4, is_reversed=False, model_type="sl", run_dir=None):
    with torch.no_grad(): 
        if model_type == "sl":
            inputs, targets, lengths, mask, cues = generate_batch(
                trial_params=TASK,
                is_reversed=is_reversed,
                cues=["A","B","B","C"],
                rewards=[1,1,0,0] if not is_reversed else [0,0,1,1])
            ys, hs = rnn.forward(inputs)
            ys_np = ys.detach().numpy()   
            plot_inputs = inputs.detach().numpy()
        else:
            # FIX: Ensure is_reversed is passed to the RL environment
            env = RLTask(batch_size=batch_size, is_reversed=is_reversed)
            stimuli_idx = [0, 1, 1, 2][:batch_size]
            cues = [['A', 'B', 'C'][s] for s in stimuli_idx]
            lengths = torch.tensor(env.seq_lens[:batch_size])
            
            inputs = torch.zeros((batch_size, env.max_seq_len, 3))
            for b in range(batch_size):
                stim_class = stimuli_idx[b]
                start = env.stim_starts[b]
                end = env.stim_ends[b]
                inputs[b, start:end, stim_class] = 1.0

            action_probs, values, _ = rnn.rl_model(inputs)
            _, hs = rnn.forward(inputs)

            # FIX: Build a 4th channel to visually represent the reward window
            dummy_channel = torch.zeros((batch_size, env.max_seq_len, 1))
            forced_rewards = [1, 1, 0, 0] if not is_reversed else [0, 0, 1, 1]
            for b in range(batch_size):
                if forced_rewards[b]:
                    dummy_channel[b, env.reward_starts[b]:env.reward_ends[b], 0] = 1.0
                    
            plot_inputs = torch.cat([inputs, dummy_channel], dim=-1).detach().numpy()

    fig, axes = plt.subplots(batch_size, 1, figsize=(10, 2 * batch_size), sharex=True)
    for i in range(batch_size):
        trial_len = lengths[i].item()

        # FIX: Use plot_inputs for both SL and RL so the reward renders
        axes[i].imshow(
            plot_inputs[i, :trial_len].T,
            aspect='auto',
            cmap='Greys',
            interpolation='nearest'
        )

        x_axis = np.arange(trial_len)

        if model_type == "sl":
            pred_seq = ys_np[i, :trial_len, 0]
            y_pred = 3 - (pred_seq * 3)
            target_seq = targets[i, :trial_len, 0].detach().numpy()       
            y_target = 3 - (target_seq * 3)

            axes[i].plot(x_axis, y_target, color='red', label='SL Target', linewidth=2)
            axes[i].plot(x_axis, y_pred, color='green', label='RNN Prediction', linestyle='--')
        else:
            critic_seq = values[i, :trial_len].squeeze().detach().numpy()
            
            # FIX: Map expected value (0.0 to 1.0) onto visual y-coordinates (3 to 0) 
            y_critic = 3 - (critic_seq * 3)
            axes[i].plot(x_axis, y_critic, color='purple', label='Critic Expected Value', linewidth=2)

        # Standardize Y-ticks
        axes[i].set_yticks([0, 1, 2, 3])
        axes[i].set_yticklabels(['A', 'B', 'C', 'R'])
        axes[i].set_ylabel(f"Trial {i}\n({cues[i]})")

    axes[-1].set_xlabel("Timestep")
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
        events=None,
        
        hs_rev=None, cues_rev=None, lengths_rev=None, inputs_rev=None, events_rev=None):
    
    
    hs_np = hs.detach().numpy()
    inputs_np = inputs.detach().numpy()
    batch_size = hs_np.shape[0]

    # Fit PCA on a snapshot to maximize stimulus separation in 2D space
    passed_pca = False
    if pca is None:
        delay_hs = []
        for i in range(batch_size):
            trial_len = lengths[i].item()
            
            if events is not None:
                t_delay_start = int(events[i]["stim_last"]) + 1
                t_delay_end = int(events[i]["reward_on"])
            else:
                trial_in_full = inputs_np[i, :trial_len, :]
                stim_activity = np.sum(trial_in_full[:, 0:3], axis=1) 
                cue_active = np.where(stim_activity > 0.5)[0]
                
                t_delay_start = int(cue_active[-1]) + 1 if len(cue_active) > 0 else max(0, trial_len - 6)
                
                if trial_in_full.shape[1] >= 4:
                    rew_active = np.where(trial_in_full[:, 3] > 0.5)[0]
                    t_delay_end = int(rew_active[0]) if len(rew_active) > 0 else trial_len - 5
                else:
                    t_delay_end = trial_len - 5
                    
            t_delay_start = min(max(0, t_delay_start), trial_len - 1)
            t_delay_end = min(max(t_delay_start, t_delay_end), trial_len)
            
            if t_delay_start < t_delay_end:
                delay_hs.append(hs_np[i, t_delay_start:t_delay_end, :])
            else:
                valid_idx = max(0, t_delay_start - 1)
                delay_hs.append(hs_np[i, valid_idx:valid_idx+1, :])
                
        delay_matrix = np.concatenate(delay_hs, axis=0)
        pca = PCA(n_components=2).fit(delay_matrix)
    else:
        print("Passed a PCA -> Cross-Projecting into provided space")
        passed_pca = True
    
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {'A': '#086ec7', 'B': '#f5a700', 'B_rew': '#f5a700', 'B_unrew': '#b37a00', 'C': '#31A354'}
  
    def _process_and_draw(d_hs, d_cues, d_lengths, d_inputs, d_events, d_is_rev, alpha_scale):
        d_hs_np = d_hs.detach().numpy()
        d_inputs_np = d_inputs.detach().numpy()
        b_size = d_hs_np.shape[0]

        conditions = ['A', 'B', 'B_rew', 'B_unrew', 'C'] 
        aligned_pca = {k: [] for k in conditions}
        # CHANGE: Track the end of the reward window
        aligned_events = {k: {'plot_start': [], 'stim_on': [], 'stim_last': [], 'cluster': [], 'reward_last': []} for k in conditions}

        for i in range(b_size):
            trial_len = d_lengths[i].item()
            cue = d_cues[i]
            
            # CHANGE: Transform the ENTIRE sequence to include the ITI tail
            trial_pca = pca.transform(d_hs_np[i]) 
            trial_in = d_inputs_np[i, :trial_len, :] 
            
            has_reward_channel = trial_in.shape[1] >= 4
            is_rewarded = False
            if has_reward_channel: 
                is_rewarded = np.any(trial_in[:, 3] > 0.5)
            
            if cue == 'B' and has_reward_channel:
                cue_key = 'B_rew' if is_rewarded else 'B_unrew'
            else:
                cue_key = cue
                
            if d_events is not None:
                t_stim_start = int(d_events[i]["stim_on"])
                t_stim_last = int(d_events[i]["stim_last"])
                t_cluster = int(d_events[i]["reward_on"]) - 1
                t_reward_last = int(d_events[i]["reward_last"]) # Grab End of Reward
            else:
                stim_activity = np.sum(trial_in[:, 0:3], axis=1) 
                cue_active = np.where(stim_activity > 0.5)[0]
                if len(cue_active) == 0: continue
                
                t_stim_start = int(cue_active[0])
                t_stim_last = int(cue_active[-1])
                
                if has_reward_channel:
                    rew_active = np.where(trial_in[:, 3] > 0.5)[0]
                    t_cluster = int(rew_active[0]) - 1 if len(rew_active) > 0 else trial_len - 6
                    t_reward_last = int(rew_active[-1]) if len(rew_active) > 0 else trial_len - 1
                else:
                    t_cluster = trial_len - 6
                    t_reward_last = trial_len - 1

            start_idx = max(0, t_stim_start - 2)
            
            # CHANGE: Dynamic end bounds based on model type
            if d_events is not None:
                # RL: Draw up to the absolute end of the hidden states (capturing the ITI tail)
                end_idx = d_hs_np.shape[1]
            else:
                # SL: Draw up to the end of the trial
                end_idx = trial_len
            
            traj_pca = trial_pca[start_idx : end_idx, :]
                    
            pad_len = max(0, (t_stim_start - 2) * -1)
            if pad_len > 0:
                traj_pca = np.vstack([np.tile(traj_pca[0, :], (pad_len, 1)), traj_pca])
                
            aligned_pca[cue_key].append(traj_pca)
            aligned_events[cue_key]['plot_start'].append(0)
            aligned_events[cue_key]['stim_on'].append(t_stim_start - start_idx + pad_len)
            aligned_events[cue_key]['stim_last'].append(t_stim_last - start_idx + pad_len)
            aligned_events[cue_key]['cluster'].append(t_cluster - start_idx + pad_len)
            aligned_events[cue_key]['reward_last'].append(t_reward_last - start_idx + pad_len)

        if d_is_rev:
            labels = {'A': 'Stim A (0%)', 'B': 'Stim B (50%)', 'B_rew': 'Stim B (50%) - Rewarded', 'B_unrew': 'Stim B (50%) - No Reward', 'C': 'Stim C (100%)'}
        else:
            labels = {'A': 'Stim A (100%)', 'B': 'Stim B (50%)', 'B_rew': 'Stim B (50%) - Rewarded', 'B_unrew': 'Stim B (50%) - No Reward', 'C': 'Stim C (0%)'}

        for cue_key in conditions:
            if not aligned_pca[cue_key]: continue
            
            min_len = min(len(t) for t in aligned_pca[cue_key])
            stacked_pca = np.array([t[:min_len] for t in aligned_pca[cue_key]])
           
            mean_pca = np.mean(stacked_pca, axis=0)
            se_pca = np.std(stacked_pca, axis=0)
            
            x_m, y_m = mean_pca[:, 0], mean_pca[:, 1]
            x_se, y_se = se_pca[:, 0], se_pca[:, 1]
            c = colors[cue_key]

            dx = np.gradient(gaussian_filter1d(x_m, sigma=2))
            dy = np.gradient(gaussian_filter1d(y_m, sigma=2))
            norm = np.sqrt(dx**2 + dy**2) + 1e-8
            nx = -dy / norm
            ny = dx / norm
            radial_sd = np.sqrt(x_se**2 + y_se**2)

            upper = np.column_stack([x_m + nx * radial_sd, y_m + ny * radial_sd])
            lower = np.column_stack([x_m - nx * radial_sd, y_m - ny * radial_sd])
            poly = np.vstack([upper, lower[::-1]])

            ax.fill(poly[:, 0], poly[:, 1], color=c, alpha=0.2 * alpha_scale, edgecolor="none", zorder=2)
            
            line_style = '--' if 'unrew' in cue_key else '-'
            line_label = labels[cue_key] if alpha_scale == 1.0 else "_nolegend_"
            ax.plot(x_m, y_m, color=c, linewidth=2.5, linestyle=line_style, alpha=alpha_scale, label=line_label, zorder=4)
            
            def median_event_idx(event_name):
                if not aligned_events[cue_key][event_name]: return 0
                idx = int(np.round(np.median(aligned_events[cue_key][event_name])))
                return int(np.clip(idx, 0, min_len - 1))

            idx_start = median_event_idx("plot_start")
            idx_stim_on = median_event_idx("stim_on")
            idx_stim_last = median_event_idx("stim_last")
            idx_cluster = median_event_idx("cluster")
            idx_reward_last = median_event_idx("reward_last")

            ax.scatter(x_m[idx_start], y_m[idx_start], color='black', marker='o', s=40, alpha=alpha_scale, zorder=6)
            ax.scatter(x_m[idx_stim_on], y_m[idx_stim_on], color='black', marker='^', s=80, alpha=alpha_scale, zorder=6)
            ax.scatter(x_m[idx_stim_last], y_m[idx_stim_last], color='black', marker='d', s=80, alpha=alpha_scale, zorder=7)
            
            rgba_color = plt.matplotlib.colors.to_rgba(c, alpha=alpha_scale)
            edge_color = plt.matplotlib.colors.to_rgba('black', alpha=alpha_scale)
            ax.scatter(x_m[idx_cluster], y_m[idx_cluster], color=rgba_color, marker='X', s=150, zorder=9, edgecolor=edge_color)
            
            # CHANGE: Draw the End of Reward marker (Square)
            face_color_white = plt.matplotlib.colors.to_rgba('white', alpha=alpha_scale)
            ax.scatter(x_m[idx_reward_last], y_m[idx_reward_last], color=face_color_white, marker='s', s=60, zorder=8, edgecolor=edge_color)
    # --- 3. DRAW DATASETS ---
    has_reversal = hs_rev is not None
    # Draw baseline (faded if overlaying, normal if not)
    _process_and_draw(hs, cues, lengths, inputs, events, is_reversed, alpha_scale=0.3 if has_reversal else 1.0)
    
    # Draw reversal on top
    if has_reversal:
        _process_and_draw(hs_rev, cues_rev, lengths_rev, inputs_rev, events_rev, d_is_rev=True, alpha_scale=1.0)
    
    # --- 4. FORMATTING & LEGENDS ---
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    plt.title(custom_title if custom_title else "Anticipatory PCA Trajectories", fontsize=14, pad=15)
    plt.xlabel(f"Principal Component 1 ({pc1_var:.1f}% variance)")
    plt.ylabel(f"Principal Component 2 ({pc2_var:.1f}% variance)")
    
    handles, legend_labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(legend_labels, handles)) # De-duplicate labels
    
    # Left Legend
    stim_legend = ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', bbox_to_anchor=(0.0, -0.15), 
                            title="Stimulus & Outcome", frameon=True)
    ax.add_artist(stim_legend) 

    # Right Legend
    phase_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Trial Start'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label='Stimulus Onset'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='black', markersize=10, label='Stimulus Offset'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markeredgecolor='black', markersize=11, label='End of Delay'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', markersize=8, label='End of Reward')
    
    ]
    ax.legend(handles=phase_elements, loc='upper right', bbox_to_anchor=(1.0, -0.15), 
              title="Phase Key", ncol=2, frameon=True)

    # Center Legend (Only if Overlay)
    if has_reversal:
        epoch_elements = [
            Line2D([0], [0], color='black', linewidth=3, alpha=1.0, label='Post-Reversal'),
            Line2D([0], [0], color='black', linewidth=3, alpha=0.3, label='Pre-Reversal (Baseline)')
        ]
        leg2 = ax.legend(handles=epoch_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), title="Learning Phase", frameon=True)
        ax.add_artist(leg2)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.subplots_adjust(bottom=0.3) # Increased from 0.25 to make room for aligned legends
    
    if run_dir:
        suffix = "overlay" if has_reversal else ("cross_pca" if passed_pca else ("reversed" if is_reversed else "normal"))
        plt.savefig(Path(run_dir) / f"pca_trajectory_{suffix}.png", dpi=300)
        plt.close()
    else:
        plt.show()

    return pca

def plot_3d_pca_mean_trajectories(
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
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    hs_np = hs.detach().numpy()
    inputs_np = inputs.detach().numpy()
    batch_size = hs_np.shape[0]

    # --- 1. FIT 3D PCA IF NOT PROVIDED ---
    passed_pca = False
    if pca is None:
        valid_hs_all = []
        for i in range(batch_size):
            trial_len = lengths[i].item()
            valid_hs_all.append(hs_np[i, :trial_len, :])
            
        joint_matrix = np.concatenate(valid_hs_all, axis=0)
        pca = PCA(n_components=3).fit(joint_matrix)
    else:
        print("Passed a PCA -> Cross-Projecting into provided 3D space")
        passed_pca = True
    
    # --- 2. DYNAMIC SPLIT AND ALIGNMENT ---
    conditions = ['A', 'B', 'B_rew', 'B_unrew', 'C'] 
    aligned_pca = {k: [] for k in conditions}
    aligned_events = {
        k: {'plot_start': [], 'stim_on': [], 'stim_last': [], 'cluster': []}
        for k in conditions
    }

    for i in range(batch_size):
        trial_len = lengths[i].item()
        cue = cues[i]
        
        trial_pca = pca.transform(hs_np[i, :trial_len, :])
        trial_in = inputs_np[i, :trial_len, :] 
        
        has_reward_channel = trial_in.shape[1] >= 4
        is_rewarded = False
        if has_reward_channel: 
            is_rewarded = np.any(trial_in[:, 3] > 0.5)
        elif events is not None:
            is_rewarded = np.random.rand() > 0.5 
            
        if cue == 'B' and (has_reward_channel or events is not None):
            cue_key = 'B_rew' if is_rewarded else 'B_unrew'
        else:
            cue_key = cue
            
        if events is not None:
            t_stim_start = int(events[i]["stim_on"])
            t_stim_last = int(events[i]["stim_last"])
            t_cluster = int(events[i]["reward_on"]) - 1
        else:
            stim_activity = np.sum(trial_in[:, 0:3], axis=1) 
            cue_active = np.where(stim_activity > 0.5)[0]
            if len(cue_active) == 0: continue

            t_stim_start = int(cue_active[0])
            t_stim_last = int(cue_active[-1])
            
            if has_reward_channel:
                rew_active = np.where(trial_in[:, 3] > 0.5)[0]
                t_cluster = int(rew_active[0]) - 1 if len(rew_active) > 0 else trial_len - 6
            else:
                t_cluster = trial_len - 6

        start_idx = max(0, t_stim_start - 2)
        end_idx = min(t_cluster + 1, trial_len)
        
        traj_pca = trial_pca[start_idx : end_idx, :]
                
        pad_len = max(0, (t_stim_start - 2) * -1)
        if pad_len > 0:
            traj_pca = np.vstack([np.tile(traj_pca[0, :], (pad_len, 1)), traj_pca])
            
        aligned_pca[cue_key].append(traj_pca)

        aligned_events[cue_key]['plot_start'].append(0)
        aligned_events[cue_key]['stim_on'].append(t_stim_start - start_idx + pad_len)
        aligned_events[cue_key]['stim_last'].append(t_stim_last - start_idx + pad_len)
        aligned_events[cue_key]['cluster'].append(t_cluster - start_idx + pad_len)
    
    # --- 3. 3D PLOTTING ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {
        'A': '#086ec7', 'B': '#f5a700', 'B_rew': '#f5a700', 'B_unrew': '#b37a00', 'C': '#31A354'
    }
    
    labels = {
        'A': f"Stim A ({'0%' if is_reversed else '100%'})", 
        'B': 'Stim B (50%)',                 
        'B_rew': 'Stim B (50%) - Rewarded', 
        'B_unrew': 'Stim B (50%) - No Reward', 
        'C': f"Stim C ({'100%' if is_reversed else '0%'})"
    }

    z_min = float('inf')
    for cue_key in conditions:
        if aligned_pca[cue_key]:
            z_min = min(z_min, np.min([np.min(t[:, 2]) for t in aligned_pca[cue_key]]))
    z_min -= 0.5 
    
    for cue_key in conditions:
        if not aligned_pca[cue_key]: continue
        
        min_len = min(len(t) for t in aligned_pca[cue_key])
        stacked_pca = np.array([t[:min_len] for t in aligned_pca[cue_key]])
       
        mean_pca = np.mean(stacked_pca, axis=0)
        
        x_m, y_m, z_m = mean_pca[:, 0], mean_pca[:, 1], mean_pca[:, 2]
        c = colors[cue_key]
        line_style = '--' if 'unrew' in cue_key else '-'
        
        ax.plot(x_m, y_m, z_m, color=c, linewidth=3, linestyle=line_style, label=labels[cue_key], zorder=4)
        ax.plot(x_m, y_m, zs=z_min, zdir='z', color=c, linewidth=1.5, linestyle=line_style, alpha=0.2, zorder=1)
        
        def median_event_idx(event_name):
            if not aligned_events[cue_key][event_name]: return 0
            idx = int(np.round(np.median(aligned_events[cue_key][event_name])))
            return int(np.clip(idx, 0, min_len - 1))

        idx_start = median_event_idx("plot_start")
        idx_stim_on = median_event_idx("stim_on")
        idx_stim_last = median_event_idx("stim_last")
        idx_cluster = median_event_idx("cluster")

        markers = [
            (idx_start, 'o', 40, 'black', 1.0),
            (idx_stim_on, '^', 80, 'black', 1.0),
            (idx_stim_last, 'd', 80, 'black', 1.0),
            (idx_cluster, 'X', 120, c, 1.0) # End of delay marker
        ]
        
        for idx, marker, size, m_color, alpha_val in markers:
            if marker == 'X':
                ax.scatter(x_m[idx], y_m[idx], z_m[idx], color=m_color, marker=marker, s=size, edgecolor='black', alpha=alpha_val, depthshade=True, zorder=6)
            else:
                ax.scatter(x_m[idx], y_m[idx], z_m[idx], color=m_color, marker=marker, s=size, alpha=alpha_val, depthshade=True, zorder=6)
            
            # Floor Shadows
            ax.scatter(x_m[idx], y_m[idx], z_min, color=m_color, marker=marker, s=size/2, alpha=0.2, zorder=1)
        
    # --- 4. FORMATTING ---
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    pc3_var = pca.explained_variance_ratio_[2] * 100
    
    plt.title(custom_title if custom_title else "Anticipatory 3D PCA Trajectories", fontsize=14, pad=15)
    ax.set_xlabel(f"PC 1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC 2 ({pc2_var:.1f}%)")
    ax.set_zlabel(f"PC 3 ({pc3_var:.1f}%)")
    ax.set_zlim(bottom=z_min)
    
    handles, legend_labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(legend_labels, handles))
    
    stim_legend = ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', bbox_to_anchor=(0.0, -0.05), 
                            title="Stimulus & Outcome", frameon=True)
    ax.add_artist(stim_legend) 
    
    phase_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Trial Start'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label='Stimulus Onset'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='black', markersize=10, label='Stimulus Offset'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markeredgecolor='black', markersize=11, label='End of Delay (Cluster)')
    ]
    ax.legend(handles=phase_elements, loc='upper right', bbox_to_anchor=(1.0, -0.05), 
              title="Phase Key", ncol=2, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    
    if run_dir:
        suffix = "cross_pca_3d" if passed_pca else ("reversed_3d" if is_reversed else "normal_3d")
        plt.savefig(Path(run_dir) / f"pca_trajectory_{suffix}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_clusters(hs, cues, lengths, is_reversed=False, run_dir=None, pca=None):
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
    if pca is None:
        pca = PCA(n_components=2)
        snapshot_pca = pca.fit_transform(snapshot_matrix)
        title_suffix = "PCA Clusters at End of Delay" if pca else ("Reversed Inputs" if is_reversed else "Normal Inputs")
    else:
        print("Passed a PCA -> Cross-Projecting into provided space")
        snapshot_pca = pca.transform(snapshot_matrix)
        title_suffix = "Cross-Projected PCA Clusters" if pca else ("Reversed Inputs" if is_reversed else "Normal Inputs")
    
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
    
    plt.title("Representational Snapshot | "+title_suffix)
    
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
    total_recorded_steps = len(live_performance['A'])
    x_axis = np.arange(total_recorded_steps)
    
    # Detect if data is epoch-level (RL) or batch-level (SL)
    is_epoch_level = total_recorded_steps < (reversal_epoch * batches_per_epoch)
    
    if is_epoch_level:
        reversal_point = reversal_epoch
        xlabel = "Training Epoch"
    else:
        reversal_point = reversal_epoch * batches_per_epoch
        xlabel = "Training Batch"
        
    plt.figure(figsize=(12, 6))
    
    # Plot the data
    plt.plot(x_axis, live_performance['A'], color='blue', alpha=0.8, linewidth=1.5, label='Stimulus A')
    plt.plot(x_axis, live_performance['B'], color='orange', alpha=0.8, linewidth=1.5, label='Stimulus B (50%)')
    plt.plot(x_axis, live_performance['C'], color='green', alpha=0.8, linewidth=1.5, label='Stimulus C')
    
    # Mark Reversal Point dynamically
    plt.axvline(x=reversal_point, color='black', linestyle='--', linewidth=2.5, label='Reversal Initiated')
    
    gamma = TRAINING.get("gamma",0.99)
    delay_duration = TASK.get("delay_duration", 15)
    stim_duration = TASK.get("stimulus_duration", 10)

    # Average distance to reward from the stimulus window
    distances = np.arange(delay_duration + 1, delay_duration + stim_duration + 1)
    avg_discount = np.mean(gamma ** distances)

    target_A = 1.0 * avg_discount
    target_B = 0.5 * avg_discount
    target_C = 0.0 * avg_discount

    # Target Guidelines
    plt.axhline(y=target_A, color='blue', linestyle=':', alpha=0.6, label=f'Discounted Target A ({target_A:.2f})')
    plt.axhline(y=target_B, color='orange', linestyle=':', alpha=0.6, label=f'Discounted Target B ({target_B:.2f})')
    plt.axhline(y=target_C, color='green', linestyle=':', alpha=0.6)
    
    ax1 = plt.gca()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Predicted Value")
    
    # Combine all data to find robust limits, ignoring NaNs or Infs
    all_vals = np.concatenate([live_performance['A'], live_performance['B'], live_performance['C']])
    valid_vals = all_vals[np.isfinite(all_vals)]
    
    if len(valid_vals) > 0:
        # Set y-limits with a buffer above the highest target or prediction
        ymax = max(np.percentile(valid_vals, 99), target_A)
        ax1.set_ylim(min(-0.1, np.min(valid_vals) - 0.1), ymax + 0.15)

    
    if not is_epoch_level:
        # Create secondary X-axis for Epochs only if we are plotting batches
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        
        epoch_ticks = np.arange(0, (total_recorded_steps // batches_per_epoch) + 1, 30)
        batch_ticks = epoch_ticks * batches_per_epoch
        ax1.set_xticks(batch_ticks)
        ax2.set_xticks(batch_ticks)
        ax2.set_xticklabels(epoch_ticks)
        ax2.set_xlabel("Training Epoch")
    else:
        # Lock the X-axis bounds for clean epoch viewing
        ax1.set_xlim(left=0, right=total_recorded_steps)
        
    plt.title("Evolution of Anticipatory Predictions", pad=20)
    ax1.legend(loc='center right')
    
    plt.tight_layout()
    if run_dir:
        suffix = "reversed" if reversed else "normal"
        plt.savefig(Path(run_dir) / f"batch_timeline_{suffix}.png", dpi=300)
        plt.close() 
    else:
        plt.show()


def add_phase_boundaries(T_pre, T_stim, T_delay, T_rew, T_post=15):
    """Draws vertical lines defining the aligned phase boundaries."""
    
    # 1. Stimulus Onset
    plt.axvline(x=T_pre, color='black', linestyle=':', linewidth=1.5)
    
    # 2. Delay Onset (Stimulus Offset)
    plt.axvline(x=T_pre + T_stim, color='black', linestyle=':', linewidth=1.5)
    
    # 3. Reward Onset (Delay Offset)
    if T_delay > 0:
        plt.axvline(x=T_pre + T_stim + T_delay, color='black', linestyle=':', linewidth=1.5)
        
    # 4. ITI Onset (Reward Offset)
    plt.axvline(x=T_pre + T_stim + T_delay + T_rew, color='black', linestyle=':', linewidth=1.5)
        
    y_max = plt.ylim()[1]
    
    # Text Labels
    plt.text(T_pre + (T_stim/2), y_max*0.95, 'Stim', ha='center', va='top', alpha=0.7)
    
    if T_delay > 0:
        plt.text(T_pre + T_stim + (T_delay/2), y_max*0.95, 'Delay', ha='center', va='top', alpha=0.7)
        
    plt.text(T_pre + T_stim + T_delay + (T_rew/2), y_max*0.95, 'Reward', ha='center', va='top', alpha=0.7)
    
    # ITI Label
    plt.text(T_pre + T_stim + T_delay + T_rew + (T_post/2), y_max*0.95, 'ITI', ha='center', va='top', alpha=0.7)

def plot_phenotype_overlay_timeline(phenotype_data, reversal_epoch, metric_name="Predicted Value", run_dir=None):
    """Overlays multiple phenotypes on a single batch timeline."""
    plt.figure(figsize=(12, 6))
    
    colors = {'A': 'blue', 'B': 'orange', 'C': 'green'}
    line_styles = {'Healthy_Baseline': '-', 'PD_Untreated': ':', 'PD_LDOPA': '--'}
    
    for pheno_name, perf in phenotype_data.items():
        style = line_styles.get(pheno_name, '-')
        x_axis = np.arange(len(perf['A']))
        
        for cue in ['A', 'B', 'C']:
            # Suppress individual labels to keep the legend clean
            plt.plot(x_axis, perf[cue], color=colors[cue], linestyle=style, 
                     alpha=0.8, linewidth=1.5, label="_nolegend_")
                     
    plt.axvline(x=reversal_epoch, color='black', linestyle='--', linewidth=2.5, label='Reversal Initiated')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='-', lw=2),
        Line2D([0], [0], color='black', linestyle=':', lw=2),
        Line2D([0], [0], color='black', linestyle='--', lw=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8)
    ]
    labels = ['Healthy', 'PD Untreated', 'PD L-DOPA', 'Stimulus A', 'Stimulus B', 'Stimulus C']
    
    plt.legend(custom_lines, labels, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.title(f"Evolution of {metric_name} - Phenotype Overlay", pad=20)
    plt.xlabel("Training Epoch")
    plt.ylabel(metric_name)
    plt.tight_layout()
    
    if run_dir:
        plt.savefig(Path(run_dir) / f"phenotype_overlay_{metric_name.replace(' ', '_')}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_variance_batch_timeline(summary_stats, reversal_epoch, metric_name="Predicted Value", run_dir=None, live_performance=None):
    """Plots a timeline with shaded standard error regions across multiple runs."""
    plt.figure(figsize=(12, 6))
    colors = {'A': 'blue', 'B': 'orange', 'C': 'green'}

    gamma = TRAINING.get("gamma", 0.99)
    delay_duration = TASK.get("delay_duration", 15)
    stim_duration = TASK.get("stimulus_duration", 10)

    distances = np.arange(delay_duration + 1, delay_duration + stim_duration + 1)
    avg_discount = np.mean(gamma ** distances)

    c_vals = live_performance['C'] if isinstance(live_performance['C'], list) else summary_stats['C']['mean']
    penalty_offset = np.mean(c_vals[:reversal_epoch])

    target_A = (1.0 * avg_discount) + penalty_offset
    target_B = (0.5 * avg_discount) + penalty_offset
    target_C = penalty_offset

    for cue, stats in summary_stats.items():
        mean = stats['mean']
        se = stats['se']
        times = np.arange(len(mean))
        color = colors.get(cue, 'black')
        
        plt.plot(times, mean, linewidth=2, label=f'Stimulus {cue}', color=color)
        plt.fill_between(times, mean - se, mean + se, color=color, alpha=0.3)


    plt.axhline(y=target_A, color='blue', linestyle=':', alpha=0.6, label=f'Target A ({target_A:.2f})')
    plt.axhline(y=target_B, color='orange', linestyle=':', alpha=0.6, label=f'Target B ({target_B:.2f})')
    plt.axhline(y=target_C, color='green', linestyle=':', alpha=0.6)
    
    plt.axvline(x=reversal_epoch, color='black', linestyle='--', linewidth=2.5, label='Reversal Initiated')
    
    plt.title(f"{metric_name} Timeline (Mean ± SE)", pad=20)
    plt.xlabel("Training Epoch")
    plt.ylabel(metric_name)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if run_dir:
        plt.savefig(Path(run_dir) / f"variance_timeline_{metric_name.replace(' ', '_')}.png", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_decoding_stimulus(accuracies_dict, avg_accuracy, T_pre, T_stim, T_delay, T_rew, T_post, reversed=False, run_dir=None):
    plt.figure(figsize=(8, 5))
    colors = {'A': 'crimson', 'B': 'forestgreen', 'C': 'royalblue'}
    
    for cue, acc_list in accuracies_dict.items():
        times = np.arange(len(acc_list))
        plt.plot(times, acc_list, marker='o', markersize=4, linewidth=2, label=f'Stimulus {cue}', color=colors.get(cue, 'black'))
        
    times = np.arange(len(avg_accuracy))
    plt.plot(times, avg_accuracy, alpha=0.4, marker='o', markersize=2, linewidth=2, label='Avg accuracy', color="black", linestyle='--')
    plt.axhline(y=1/3, color='gray', linestyle='--', label='Chance (33%)')

    add_phase_boundaries(T_pre, T_stim, T_delay, T_rew)

    plt.title("Decoding Accuracy by Stimulus Identity")
    plt.xlabel("Aligned Time Step")
    plt.ylabel("Accuracy (True Positive Rate)")
    plt.ylim(0, 1.05)
    plt.legend(loc='lower left') # Moved to avoid blocking post-ITI curve drop
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if run_dir:
        suffix = "reversed" if reversed else "normal"
        plt.savefig(Path(run_dir) / f"decoding_stimulus_{suffix}.png", dpi=300)
        plt.close()
    else:
        plt.show()



def plot_decoding_trajectories(mean_predictions, T_pre, T_stim, T_delay, T_rew, T_post, reversed=False, run_dir=None):
    plt.figure(figsize=(8, 5))
    colors = {'A': 'crimson', 'B': 'forestgreen', 'C': 'royalblue'}
    
    for cue, trajectory in mean_predictions.items():
        times = np.arange(len(trajectory))
        plt.plot(times, trajectory, linewidth=2, label=f'Cue {cue}', color=colors.get(cue, 'black'))

    add_phase_boundaries(T_pre, T_stim, T_delay, T_rew)

    plt.title("Decoded Behavioral Trajectories (Predicted Value/Lick Rate)")
    plt.xlabel("Aligned Time Step")
    plt.ylabel("Decoded Output Value")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if run_dir:
        suffix = "reversed" if reversed else "normal"
        plt.savefig(Path(run_dir) / f"decoding_trajectory_{suffix}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_aligned_model_outputs(mean_outputs, T_pre, T_stim, T_delay, T_rew, T_post, reversed=False, run_dir=None):
    plt.figure(figsize=(8, 5))
    colors = {'A': 'crimson', 'B': 'forestgreen', 'C': 'royalblue'}
    
    for cue, trajectory in mean_outputs.items():
        times = np.arange(len(trajectory))
        plt.plot(times, trajectory, linewidth=2.5, linestyle='--', label=f'Direct Output {cue}', color=colors.get(cue, 'black'))

    add_phase_boundaries(T_pre, T_stim, T_delay, T_rew)

    plt.title("Direct Model Output Probabilities over Aligned Trials")
    plt.xlabel("Aligned Time Step")
    plt.ylabel("Predicted Expected Value (or Lick Probability)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if run_dir:
        suffix = "reversed" if reversed else "normal"
        plt.savefig(Path(run_dir) / f"model_outputs_{suffix}.png", dpi=300)
        plt.close()
    else:
        plt.show()

# --- Update Execution Sequence ---
def run_decoding_analysis(model, trial_params, run_dir, baseline_ckpt="checkpoints/weights_baseline.pth", reversal_ckpt="checkpoints/weights_final_reversal.pth", model_type="sl"):
    print("--- Starting Pre-Reversal Decoding Analysis ---")
    model.load_state_dict(torch.load(baseline_ckpt))

    acc_dict, avg_acc, T_pre, T_stim, T_delay, T_rew, T_post = train_stimulus_decoders(model, trial_params, reversed=False, model_type=model_type)
    plot_decoding_stimulus(acc_dict, avg_acc, T_pre, T_stim, T_delay, T_rew, T_post, reversed=False, run_dir=run_dir)

    mean_preds, _, _, _, _, _ = train_continuous_decoders(model, trial_params, reversed=False, model_type=model_type)
    plot_decoding_trajectories(mean_preds, T_pre, T_stim, T_delay, T_rew, T_post, reversed=False, run_dir=run_dir)

    mean_outputs, _, _, _, _, _ = get_aligned_model_outputs(model, trial_params, is_reversed=False, model_type=model_type)
    plot_aligned_model_outputs(mean_outputs, T_pre, T_stim, T_delay, T_rew, T_post, reversed=False, run_dir=run_dir)
    
    print("--- Starting Post-Reversal Decoding Analysis ---")
    model.load_state_dict(torch.load(reversal_ckpt))
    
    acc_dict, avg_acc, T_pre, T_stim, T_delay, T_rew, T_post = train_stimulus_decoders(model, trial_params, reversed=True, model_type=model_type)
    plot_decoding_stimulus(acc_dict, avg_acc, T_pre, T_stim, T_delay, T_rew, T_post, reversed=True, run_dir=run_dir)

    mean_preds, _, _, _, _, _ = train_continuous_decoders(model, trial_params, reversed=True, model_type=model_type)
    plot_decoding_trajectories(mean_preds, T_pre, T_stim, T_delay, T_rew, T_post, reversed=True, run_dir=run_dir)

    mean_outputs, _, _, _, _, _ = get_aligned_model_outputs(model, trial_params, is_reversed=True, model_type=model_type)
    plot_aligned_model_outputs(mean_outputs, T_pre, T_stim, T_delay, T_rew, T_post, reversed=True, run_dir=run_dir)
    
    print("Decoding Analysis Complete.")

# * Note: You must also ensure that everywhere you call your extraction functions in plotting.py, 
# you unpack the newly added `ys` parameter at the end of the tuple. Example:
# hs, cues, lengths, inputs, events, ys = extract_hidden_states_rl(...)
# hs, cues, lengths, inputs, targets, ys = extract_hidden_states(...)

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


def run_baseline_plots(cfg, model_type="sl", run_dir=None):
    if model_type == "rl":
        rnn = RLModelWrapper(ActorCriticRNN(input_size=3, hidden_size=64, num_actions=2))
    else:
        rnn = ScratchRNN()
    rnn.load_state_dict(torch.load("checkpoints/weights_baseline.pth"))

    if model_type == "rl":
        hs, cues, lengths, inputs, events, _ = extract_hidden_states_rl(
            model=rnn,
            trial_params=TASK,
            num_trials_per_stim=cfg.num_trials_per_stim
        )
    else:
        hs, cues, lengths, inputs, targets, _ = extract_hidden_states(
            model=rnn,
            trial_params=TASK,
            num_trials_per_stim=cfg.num_trials_per_stim,
            is_reversed=False
        )
        events = None

    print("plotting baseline trajectories ")
    fitted_pca = plot_pca_mean_trajectories(
        hs=hs, 
        cues=cues, 
        lengths=lengths, 
        inputs=inputs, 
        is_reversed=False, 
        run_dir=run_dir,
        custom_title=f"Standard Mean PCA Trajectories (Pre-Reversal)",
        events=events
    )
    plot_3d_pca_mean_trajectories(
        hs=hs, 
        cues=cues, 
        lengths=lengths, 
        inputs=inputs, 
        is_reversed=False, 
        run_dir=run_dir,
        custom_title=f"Standard 3D PCA Trajectories (Pre-Reversal)",
        events=events
    )
    
    plot_clusters(hs, cues, lengths, is_reversed=False, run_dir=run_dir, pca=fitted_pca)
    plot_distance_matrix(hs, cues, lengths, is_reversed=False, run_dir=run_dir)
    plot_predictions(rnn, is_reversed=False, model_type=model_type, run_dir=run_dir)

def run_final_reversal_plots(cfg, model_type="sl", run_dir=None):
    if model_type == "rl":
        rnn = RLModelWrapper(ActorCriticRNN(input_size=3, hidden_size=64, num_actions=2))
    else:
        rnn = ScratchRNN()
    rnn.load_state_dict(torch.load("checkpoints/weights_final_reversal.pth"))

    if model_type == "rl":
        hs, cues, lengths, inputs, events, _ = extract_hidden_states_rl(
            model=rnn,
            trial_params=TASK,
            num_trials_per_stim=cfg.num_trials_per_stim,
            is_reversed=True
        )
    else:
        hs, cues, lengths, inputs, targets, _ = extract_hidden_states(
            model=rnn,
            trial_params=TASK,
            num_trials_per_stim=cfg.num_trials_per_stim,
            is_reversed=True
        )
        events = None
    print("plotting final reversal trajectories ")
    fitted_pca = plot_pca_mean_trajectories(
        hs=hs, 
        cues=cues, 
        lengths=lengths, 
        inputs=inputs,  
        is_reversed=True,
        run_dir=run_dir,
        custom_title=f"Standard Mean PCA Trajectories (Post-Reversal)",
        events=events
    )
    plot_clusters(hs, cues, lengths, is_reversed=True, run_dir=run_dir, pca=fitted_pca)
    plot_distance_matrix(hs, cues, lengths, is_reversed=True, run_dir=run_dir)
    
    plot_predictions(rnn, is_reversed=True, model_type=model_type, run_dir=run_dir)


def run_cross_projection_pca(
        model, 
        trial_params, 
        baseline_ckpt, 
        reversal_ckpt, 
        run_dir=None,
        model_type="sl"):

    print("--- Running Joint PCA (Baseline + Reversal Overlay) ---")
    
    # 1. Extract Baseline Data
    model.load_state_dict(torch.load(baseline_ckpt))
    if model_type == "rl":
        hs_base, cues_base, lengths_base, inputs_base, events_base, _ = extract_hidden_states_rl(
            model=model, trial_params=trial_params, num_trials_per_stim=100
        )
    else:
        hs_base, cues_base, lengths_base, inputs_base, _, _ = extract_hidden_states(
            model=model, trial_params=trial_params, is_reversed=False
        )
        events_base = None

    # 2. Extract Reversal Data
    model.load_state_dict(torch.load(reversal_ckpt))
    if model_type == "rl":
        hs_rev, cues_rev, lengths_rev, inputs_rev, events_rev, _ = extract_hidden_states_rl(
            model=model, trial_params=trial_params, num_trials_per_stim=100, is_reversed=True
        )
    else:
        hs_rev, cues_rev, lengths_rev, inputs_rev, _, _ = extract_hidden_states(
            model=model, trial_params=trial_params, is_reversed=True
        )
        events_rev = None

    # 3. Fit Joint PCA on the ENTIRE sequence of BOTH models
    valid_hs_all = []
    
    # Add Baseline sequences
    for i in range(len(cues_base)):
        trial_len = lengths_base[i].item()
        valid_hs_all.append(hs_base[i, :trial_len, :].detach().numpy())
        
    # Add Reversal sequences
    for i in range(len(cues_rev)):
        trial_len = lengths_rev[i].item()
        valid_hs_all.append(hs_rev[i, :trial_len, :].detach().numpy())

    joint_matrix = np.concatenate(valid_hs_all, axis=0)
    
    # Fit the shared space
    joint_pca = PCA(n_components=2)
    joint_pca.fit(joint_matrix)
    
    # 4. Plot overlaid cross-projection
    print("Plotting Overlaid Cross Projection...")
    plot_pca_mean_trajectories(
        hs=hs_base, 
        cues=cues_base, 
        lengths=lengths_base, 
        inputs=inputs_base,
        events=events_base,
        hs_rev=hs_rev,
        cues_rev=cues_rev,
        lengths_rev=lengths_rev,
        inputs_rev=inputs_rev,
        events_rev=events_rev,
        pca=joint_pca,
        run_dir=run_dir,
        custom_title="Joint PCA Projection (Baseline + Reversal Overlay)"
    )

def run_aligned_decoding_distance_experiment(reward_percentages=[0.0, 0.25, 0.5, 0.75, 1.0], epochs=80, run_dir=None):
    # Setup fixed alignment parameters for the plot
    T_pre = 5
    T_stim = TASK["stimulus_duration"]
    T_delay = TASK["delay_duration"]
    T_rew = TASK["reward_duration"]
    T_total = T_pre + T_stim + T_delay + T_rew
    
    distances_over_time = {}
    
    print("Starting Aligned Decoding Accuracy Distance Experiment...")
    
    for pct in reward_percentages:
        print(f"\n--- Training SL Model & Decoders: B Reward Probability = {int(pct*100)}% ---")
        
        # 1. Modify task configuration
        local_task = TASK.copy()
        local_task["reward_probs"] = {"A": 1.0, "B": pct, "C": 0.0}
        
        # 2. Train SL model
        dataset = generate_full_dataset(
            epochs=epochs, trial_params=local_task,
            trial_counts={"A": 20, "B": 20, "C": 20}, 
            batches_per_epoch=20, reversal_epoch=999, SIGMA=1.2
        )
        rnn = ScratchRNN(input_size=4, hidden_size=128, output_size=1)
        trained_rnn, _, _ = train_model(
            rnn=rnn, dataset=dataset, lr=1e-3, 
            epochs=epochs, batches_per_epoch=20, batch_size=32
        )
        
        # 3. Extract evaluation dataset (100 trials per stim for clean decoding)
        print("Extracting and aligning hidden states...")
        hs, cues, lengths, inputs, targets, _ = extract_hidden_states(
            model=trained_rnn, trial_params=local_task, num_trials_per_stim=100, is_reversed=False
        )
        
        hs_np = hs.cpu().numpy()
        inputs_np = inputs.cpu().numpy()
        cues_np = np.array(cues)
        batch_size = hs_np.shape[0]
        
        # ==========================================
        # 4. ALIGN TRIALS BY STIMULUS ONSET
        # ==========================================
        aligned_hs = np.zeros((batch_size, T_total, hs_np.shape[2]))
        
        for i in range(batch_size):
            # Find exact timestep the stimulus turns on for this specific trial
            stim_mask = np.any(inputs_np[i, :, :3] > 0.5, axis=1)
            onset = np.argmax(stim_mask) if np.any(stim_mask) else 5
            
            start_idx = onset - T_pre
            end_idx = onset + T_stim + T_delay + T_rew
            
            trial_hs = hs_np[i]
            
            # Slice and pad so every trial starts exactly T_pre steps before onset
            if start_idx < 0:
                pad = np.tile(trial_hs[0], (-start_idx, 1))
                slice_part = trial_hs[0 : min(len(trial_hs), end_idx)]
                aligned = np.vstack([pad, slice_part])
            else:
                aligned = trial_hs[start_idx : min(len(trial_hs), end_idx)]
                
            # If the trial ended before the reward window finished, pad the tail
            if len(aligned) < T_total:
                pad = np.tile(aligned[-1], (T_total - len(aligned), 1))
                aligned = np.vstack([aligned, pad])
                
            aligned_hs[i] = aligned[:T_total]
        
        # ==========================================
        # 5. DECODE OVER ALIGNED TIMESTEPS
        # ==========================================
        print("Training decoders on aligned timesteps...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_A = []
        acc_B = []
        
        for t in range(T_total):
            X_t = aligned_hs[:, t, :]
            fold_A, fold_B = [], []
            
            for train_idx, test_idx in cv.split(X_t, cues_np):
                X_train, X_test = X_t[train_idx], X_t[test_idx]
                y_train, y_test = cues_np[train_idx], cues_np[test_idx]
                
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred, labels=['A', 'B', 'C'])
                sums = cm.sum(axis=1)
                
                # Extract individual class accuracies safely
                a_val = cm[0, 0] / sums[0] if sums[0] > 0 else 0
                b_val = cm[1, 1] / sums[1] if sums[1] > 0 else 0
                
                fold_A.append(a_val)
                fold_B.append(b_val)
                
            acc_A.append(np.mean(fold_A))
            acc_B.append(np.mean(fold_B))
            
        # Calculate the absolute difference between their accuracies
        distances_over_time[pct] = np.abs(np.array(acc_A) - np.array(acc_B))

    # ==========================================
    # 6. PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(reward_percentages)))
    
    for idx, pct in enumerate(reward_percentages):
        plt.plot(np.arange(T_total), distances_over_time[pct], 
                 label=f'B Reward = {int(pct*100)}%', color=colors[idx], linewidth=2.5)
        
    # Draw ALIGNED phase boundaries
    plt.axvline(x=T_pre, color='black', linestyle=':', alpha=0.7)
    plt.axvline(x=T_pre + T_stim, color='black', linestyle=':', alpha=0.7)
    plt.axvline(x=T_pre + T_stim + T_delay, color='black', linestyle=':', alpha=0.7)
    
    plt.text(T_pre + (T_stim/2), plt.ylim()[1]*0.95, 'Stimulus', ha='center', va='top', alpha=0.7)
    plt.text(T_pre + T_stim + (T_delay/2), plt.ylim()[1]*0.95, 'Delay', ha='center', va='top', alpha=0.7)
    plt.text(T_pre + T_stim + T_delay + (T_rew/2), plt.ylim()[1]*0.95, 'Reward', ha='center', va='top', alpha=0.7)

    plt.title("Difference in Decoding Accuracy (|Acc_A - Acc_B|) over Time")
    plt.xlabel("Aligned Timesteps (Locked to Stimulus Onset)")
    plt.ylabel("Accuracy Difference")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="Target Probability of B", loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if run_dir:
        plt.savefig(Path(run_dir) / "aligned_decoding_distance.png", dpi=300)
        plt.close()
    else:
        plt.show()
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

def plot_all_graphs(train=False, model_type="sl", plots=None, run_dir=None, condition = "Healthy_Baseline"):
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
        PHENOTYPES = TRAINING.get("phenotypes")
        if model_type == "rl":
            gamma = TRAINING.get("gamma",0.99)
            virtual_epoch_length = 1000
            total_time = cfg.epochs * virtual_epoch_length

            alpha_plus = PHENOTYPES[condition]["alpha_plus"]
            alpha_minus = PHENOTYPES[condition]["alpha_minus"]
            _, _, live_performance, live_actor_performance = train_rl_model(
                total_timesteps=total_time, batch_size=cfg.batch_size, lr=cfg.lr, gamma=gamma,
                alpha_plus=alpha_plus,
                alpha_minus=alpha_minus

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
            run_dir=run_dir,
            model_type=model_type
        )
    if "heatmap" in plots:
        if model_type == "sl":
            df_results= run_hyperparameter_search()
            plot_heatmap(df_results, param1='hidden_size', param2='target_sigma')
        elif model_type == "rl":
            df_results= run_rl_hyperparameter_search()
            plot_rl_convergence_heatmap(df_results, param1='bptt_horizon', param2='gamma')
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
    if "reward_percentage" in plots:
        run_aligned_decoding_distance_experiment(
            reward_percentages=[0.0, 0.25, 0.5, 0.75, 1.0], 
            epochs=80, 
            run_dir=run_dir
        )
if __name__ == "__main__":
    # 1. Ask user which framework model to isolate
    model_choice = input("Choose model framework to plot (sl / rl): ").strip().lower()
    if model_choice not in ["sl", "rl"]:
        model_choice = "sl" # Default fallback
        
    save = input("Save graphs? (y/n): ").lower()
    current_run_dir = initialize_run_directory() if save == "y" else None
    
    # Run the setup cleanly using the choice string
    plot_all_graphs(train=True, model_type=model_choice, plots=["timeline","baseline","final_reversal","decoding"], run_dir=current_run_dir,condition="Healthy_Baseline")
