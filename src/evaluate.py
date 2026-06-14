import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from task import *  #
from models import ScratchRNN
from train import train_model, train_rl_model
from configs import TRAINING, TASK   
import os



def run_hyperparameter_search():
   # Hyperparameter ranges to try out
    hyperparameter_grid = {
        'learning_rate': [1e-3,1e-5], #[1e-3, 1e-4, 1e-5], 
        'hidden_size': [128,256],#[128, 256], 
        'target_sigma': [1.2] 
    }
    
    # Get constant parameters
    epochs = TRAINING["epochs"]
    batches_per_epoch = TRAINING["batches_per_epoch"]
    batch_size = TRAINING["batch_size"]
    reversal_epoch = TRAINING["reversal_epoch"] # Ensure this is pulled from config

    # Generates all combinations of hyperparameters
    keys, values = zip(*hyperparameter_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total experiments to run: {len(experiments)}")

    best_loss = float('inf')
    best_params = None
    all_results = []

    # Search loop
    for i, params in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)} ---")
        print(f"Parameters: {params}")

        SIGMA = params["target_sigma"]

        
        dataset = generate_full_dataset(
            epochs=epochs,
            trial_params=TASK,            
            trial_counts={"A": 10, "B": 10, "C": 12}, # Ensure this sums to desired batch_size
            batches_per_epoch=batches_per_epoch,
            reversal_epoch=reversal_epoch,
            SIGMA=SIGMA,
        )

        # Train the model with selected parameters
        rnn = ScratchRNN(input_size=4, hidden_size=params['hidden_size'], output_size=1)
        
        #CALL TRAIN MODEL WITH EXPLICIT KEYWORD ARGUMENTS
        trained_rnn, loss_history, live_performance = train_model(
            rnn=rnn, 
            dataset=dataset, 
            lr=params["learning_rate"], 
            epochs=epochs, 
            batches_per_epoch=batches_per_epoch, 
            batch_size=batch_size
        )

        convergence_epoch = -1
        tolerance = 0.1
        for e in range(len(live_performance['A'])):
            a_conv = abs(live_performance['A'][e] - 1.0) < tolerance
            b_conv = abs(live_performance['B'][e] - 0.5) < tolerance
            c_conv = abs(live_performance['C'][e] - 0.0) < tolerance
            
            if a_conv and b_conv and c_conv:
                # Check if it STAYS converged for a few batches (e.g., 5)
                # (Add logic here to look ahead, or just take the first instance)
                convergence_epoch = e
                break

        run_data = params.copy()
        run_data['convergence_epoch'] = convergence_epoch

        # Take an average of the last 3 epochs' loss
        final_loss = np.mean(loss_history[-3:])

        # Save results of this specific run
        
        run_data['final_loss'] = final_loss
        run_data['loss_history'] = loss_history
        all_results.append(run_data)

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params
            
    df_results = pd.DataFrame(all_results)

    print("\n==========================================")
    print(f"Hyperparameter Search Complete")
    print(f"Best Loss: {best_loss:.5f}")
    print(f"Best Parameters: {best_params}")
    
    return df_results

def run_rl_hyperparameter_search():
    hyperparameter_grid = {
        'gamma': [0.95,0.97,0.99,1], #[0.75, 0.8, 0.85, 0.9, 0.95],          
        'bptt_horizon': [30, 40,50,60],#[30,40,50,60],       
        'learning_rate': [1e-3], 
        'entropy_coef': [0.01], 
        'critic_coef': [2.0]          
    }
    
    epochs = TRAINING.get("epochs", 2000)
    batches_per_epoch = TRAINING["batches_per_epoch"]
    batch_size = TRAINING["batch_size"]

    keys, values = zip(*hyperparameter_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total RL experiments to run: {len(experiments)}")
    all_results = []

    for i, params in enumerate(experiments):
        print(f"\n--- Running RL Experiment {i+1}/{len(experiments)} ---")
        print(f"Parameters: {params}")

        run_save_dir = f"checkpoints/rl_search_bptt_gamma_{i}"
        os.makedirs(run_save_dir, exist_ok=True)

        # --- NEW: Calculate exact theoretical targets for this specific Gamma ---
        gamma = params["gamma"]
        
        
        stim_dur = TASK.get("stimulus_duration", 10)
        delay_dur = TASK.get("delay_duration", 0)

        distances = np.arange(delay_dur + 1, delay_dur + stim_dur + 1) 
        
        avg_discount_factor = np.mean(gamma ** distances)

        target_a = 1.0 * avg_discount_factor
        target_b = 0.5 * avg_discount_factor
        target_c = 0.0 * avg_discount_factor # Optimal action is to not lick, avoiding the -0.01 penalty
        
        print(f"Theoretical Value Targets -> A: {target_a:.3f}, B: {target_b:.3f}, C: {target_c:.3f}")

        model, loss_history, live_performance = train_rl_model(
            total_timesteps=(epochs * 1000), 
            bptt_horizon=params["bptt_horizon"], 
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            lr=params["learning_rate"],
            gamma=params["gamma"],
            entropy_coef=params["entropy_coef"],
            save_dir=run_save_dir,
            critic_coef=params["critic_coef"]
        )

        # --- NEW: Exact Value Convergence Logic ---
        convergence_epoch = -1
        total_tracked_epochs = len(live_performance['A'])
        
        # Determine tolerance dynamically based on the scale of Gamma to prevent impossible standards
        # (e.g., if target_a is 0.1, a 0.05 tolerance is huge. If it's 1.0, 0.05 is strict)
        tolerance = max(0.02, target_a * 0.1) 

        for e in range(total_tracked_epochs):
            # Check if all predictions are within the required margin of the true mathematical target
            a_conv = abs(live_performance['A'][e] - target_a) < tolerance
            b_conv = abs(live_performance['B'][e] - target_b) < tolerance
            c_conv = abs(live_performance['C'][e] - target_c) < tolerance
            
            if a_conv and b_conv and c_conv:
                # Prove it wasn't a fluke by checking stability over the next 5 epochs
                if e + 5 < total_tracked_epochs:
                    stable = True
                    for check_e in range(e, e + 5):
                        if not (abs(live_performance['A'][check_e] - target_a) < tolerance and
                                abs(live_performance['B'][check_e] - target_b) < tolerance and
                                abs(live_performance['C'][check_e] - target_c) < tolerance):
                            stable = False
                            break
                    
                    if stable:
                        convergence_epoch = e
                        break

        run_data = params.copy()
        run_data['final_loss'] = np.mean(loss_history[-10:]) if loss_history else float('inf')
        run_data['convergence_epoch'] = convergence_epoch if convergence_epoch != -1 else "Did Not Converge"
        
        all_results.append(run_data)

    df_results = pd.DataFrame(all_results)
    print("\n==========================================")
    print("RL BPTT/Gamma Search Complete")
    return df_results

def extract_hidden_states(model, trial_params, num_trials_per_stim=100, is_reversed=False, noise_stdev=0.1):
    model.eval()
    cues = ["A"] * num_trials_per_stim + ["B"] * num_trials_per_stim + ["C"] * num_trials_per_stim
    
    with torch.no_grad():
        inputs, targets, lengths, mask, cues = generate_batch(
            trial_params=trial_params, is_reversed=is_reversed, cues=cues, noise_stdev=noise_stdev
        )
        ys, hs = model(inputs)
        
    return hs, cues, lengths, inputs, targets, ys

def extract_hidden_states_rl(model, trial_params, num_trials_per_stim=100, is_reversed=False):
    model.eval()

    stimuli = np.array(
        [0] * num_trials_per_stim +
        [1] * num_trials_per_stim +
        [2] * num_trials_per_stim
    )

    

    env = RLTask(
        batch_size=len(stimuli),
        stimulus_duration=trial_params.get("stimulus_duration", 10),
        delay_duration=trial_params.get("delay_duration", 0),  # This stops the collapse!
        reward_duration=trial_params.get("reward_duration", 5),
        is_reversed=is_reversed
    )

    inputs, stimuli, lengths, events = env.get_batch(
        stimuli=stimuli,
        return_events=True
    )

    cue_names = np.array(["A", "B", "C"])
    cues = [cue_names[s] for s in stimuli]

    with torch.no_grad():
        # BEFORE the stimulus hits, avoiding "cold-start" trajectory hooks.
        warmup_len = 15
        warmup_inputs = torch.zeros((inputs.size(0), warmup_len, 3), device=inputs.device)
        
        # dd a cooldown tail so the next ITI is captured 
        cooldown_len = 25
        cooldown_inputs = torch.zeros((inputs.size(0), cooldown_len, 3), device=inputs.device)
        
        # Concatenate warmup, actual trial, and cooldown ITI
        padded_inputs = torch.cat([warmup_inputs, inputs, cooldown_inputs], dim=1)
        
        ys_padded, hs_padded = model(padded_inputs)
        
        # Strip the warmup frames so the extracted states align with the trial
        hs = hs_padded[:, warmup_len:, :]
        ys = ys_padded[:, warmup_len:, :]
    return hs, cues, lengths, inputs, events, ys

def train_stimulus_decoders(model, trial_params, reversed=False, noise_stdev=0.1, model_type="sl"): 
    if model_type == "rl":
        hs, cues, lengths, inputs, events, ys = extract_hidden_states_rl(model, trial_params, is_reversed=reversed)
    else:
        hs, cues, lengths, inputs, targets, ys = extract_hidden_states(model, trial_params, is_reversed=reversed, noise_stdev=noise_stdev)

    X_all = hs.cpu().numpy()
    y_all_1d = np.array(cues)
    inputs_np = inputs.cpu().numpy()

    T_stim = trial_params.get("stimulus_duration", 10)
    T_delay = trial_params.get("delay_duration", 0)
    T_rew = trial_params.get("reward_duration", 5)
    
    aligned_hs = align_trials(X_all, inputs_np, T_pre=3, T_stim=T_stim, T_delay=T_delay, T_rew=T_rew, T_post=15)
    batch_size, T_total, hidden_dim = aligned_hs.shape

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    per_stimulus_accuracies = {'A': [], 'B': [], 'C': []}
    avg_accuracy = []
    
    for t in range(T_total):
        X_t = aligned_hs[:, t, :]
        fold_A, fold_B, fold_C = [], [], []
        
        for train_idx, test_idx in cv.split(X_t, y_all_1d):
            X_train, X_test = X_t[train_idx], X_t[test_idx]
            y_train, y_test = y_all_1d[train_idx], y_all_1d[test_idx]
            
            decoder = LogisticRegression(max_iter=1000)
            decoder.fit(X_train, y_train)
            y_pred = decoder.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred, labels=['A', 'B', 'C'])
            class_accs = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
            fold_A.append(class_accs[0])
            fold_B.append(class_accs[1])
            fold_C.append(class_accs[2])    
            
        per_stimulus_accuracies['A'].append(np.mean(fold_A))
        per_stimulus_accuracies['B'].append(np.mean(fold_B))
        per_stimulus_accuracies['C'].append(np.mean(fold_C))
        avg_accuracy.append((np.mean(fold_A) + np.mean(fold_B) + np.mean(fold_C)) / 3)
    
    return per_stimulus_accuracies, avg_accuracy, 3, T_stim, T_delay, T_rew, 15

def train_continuous_decoders(model, trial_params, reversed=False, noise_stdev=0.1, model_type="sl", decoder_type="ridge"):
    if model_type == "rl":
        hs, cues, lengths, inputs, events, ys = extract_hidden_states_rl(model, trial_params, is_reversed=reversed)
    else:
        hs, cues, lengths, inputs, targets, ys = extract_hidden_states(model, trial_params, is_reversed=reversed, noise_stdev=noise_stdev)

    X_all = hs.cpu().numpy()
    inputs_np = inputs.cpu().numpy()
    
    # Target model's actual outputs instead of ground truth
    y_all = ys.squeeze(-1).cpu().numpy() if ys.dim() == 3 else ys.cpu().numpy()
        
    T_stim = trial_params.get("stimulus_duration", 10)
    T_delay = trial_params.get("delay_duration", 0)
    T_rew = trial_params.get("reward_duration", 5)
    
    aligned_hs = align_trials(X_all, inputs_np, T_pre=3, T_stim=T_stim, T_delay=T_delay, T_rew=T_rew, T_post=15)
    aligned_ys = align_trials(np.expand_dims(y_all, -1), inputs_np, T_pre=3, T_stim=T_stim, T_delay=T_delay, T_rew=T_rew, T_post=15)
    
    batch_size, T_total, hidden_dim = aligned_hs.shape
    predictions = np.zeros((batch_size, T_total))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for t in range(T_total):
        X_t = aligned_hs[:, t, :]
        y_t = aligned_ys[:, t]
        
        for train_idx, test_idx in cv.split(X_t, y_t):
            X_train, X_test = X_t[train_idx], X_t[test_idx]
            y_train, y_test = y_t[train_idx].ravel(), y_t[test_idx].ravel()
            
            if decoder_type == "ridge":
                decoder = Ridge(alpha=1.0)
            elif decoder_type == "lasso":
                decoder = Lasso(alpha=0.01) # Small alpha to prevent flattening everything
            elif decoder_type == "ols":
                decoder = LinearRegression()
            elif decoder_type == "rf":
                decoder = RandomForestRegressor(n_estimators=20, max_depth=3)
            else:
                raise ValueError(f"Unknown decoder_type: {decoder_type}")
            
            decoder.fit(X_train, y_train)
            predictions[test_idx, t] = decoder.predict(X_test).ravel()
            
    cues_array = np.array(cues)
    mean_predictions = {}
    for cue in ['A', 'B', 'C']:
        cue_indices = np.where(cues_array == cue)[0]
        mean_predictions[cue] = predictions[cue_indices, :].mean(axis=0)
        
    return mean_predictions, 3, T_stim, T_delay, T_rew, 15

# --- New Lick Probability Extractor ---
def get_aligned_model_outputs(model, trial_params, is_reversed=False, model_type="sl"):
    """Extracts and averages the direct outputs of the model without decoding."""
    if model_type == "rl":
        _, cues, _, inputs, _, ys = extract_hidden_states_rl(model, trial_params, is_reversed=is_reversed)
    else:
        _, cues, _, inputs, _, ys = extract_hidden_states(model, trial_params, is_reversed=is_reversed)
        
    y_all = ys.squeeze(-1).detach().cpu().numpy() if ys.dim() == 3 else ys.detach().cpu().numpy()
    inputs_np = inputs.cpu().numpy()
    
    T_stim = trial_params.get("stimulus_duration", 10)
    T_delay = trial_params.get("delay_duration", 0)
    T_rew = trial_params.get("reward_duration", 5)
    
    aligned_ys = align_trials(np.expand_dims(y_all, -1), inputs_np, T_pre=3, T_stim=T_stim, T_delay=T_delay, T_rew=T_rew, T_post=15)
    
    cues_array = np.array(cues)
    mean_outputs = {}
    for cue in ['A', 'B', 'C']:
        idx = np.where(cues_array == cue)[0]
        mean_outputs[cue] = aligned_ys[idx, :].mean(axis=0) if len(idx) > 0 else np.zeros(aligned_ys.shape[1])
        
    return mean_outputs, 3, T_stim, T_delay, T_rew, 15
def align_trials(
        data,
        inputs_np,
        T_pre=3,
        T_stim=10,
        T_delay=0,
        T_rew=5,
        T_post=15
):
    batch_size = data.shape[0]
    T_total = T_pre + T_stim + T_delay + T_rew + T_post
    
    # Check if data is 2D (batch, seq) or 3D (batch, seq, feat)
    is_3d = len(data.shape) == 3
    feature_dim = data.shape[-1] if is_3d else 1
    aligned_data = np.zeros((batch_size, T_total, feature_dim)) if is_3d else np.zeros((batch_size, T_total))
    
    for i in range(batch_size):
        # Find stimulus onset
        stim_mask = np.any(inputs_np[i, :, :3] > 0.5, axis=1)
        onset = np.argmax(stim_mask) if np.any(stim_mask) else T_pre
        
        start_idx = onset - T_pre
        end_idx = onset + T_stim + T_delay + T_rew + T_post
        trial_data = data[i]
        
        # Slice and Pad
        if start_idx < 0:
            pad = np.tile(trial_data[0], (-start_idx, 1)) if is_3d else np.tile(trial_data[0], -start_idx)
            slice_part = trial_data[0 : min(len(trial_data), end_idx)]
            aligned = np.vstack([pad, slice_part]) if is_3d else np.concatenate([pad, slice_part])
        else:
            aligned = trial_data[start_idx : min(len(trial_data), end_idx)]
            
        if len(aligned) < T_total:
            pad_len = T_total - len(aligned)
            pad = np.tile(aligned[-1], (pad_len, 1)) if is_3d else np.tile(aligned[-1], pad_len)
            aligned = np.vstack([aligned, pad]) if is_3d else np.concatenate([aligned, pad])
            
        aligned_data[i] = aligned[:T_total]
        
    return aligned_data if is_3d else aligned_data.squeeze()

def calculate_phenotype_metrics(model, live_critic, live_actor, trial_params):
    """Calculates quantitative 'breakage' metrics for Pre-Reversal and Post-Reversal phases."""
    metrics = {}
    
    reversal_epoch = TRAINING.get("reversal_epoch", 200)
    
    # ==========================================
    # 1. PRE-REVERSAL METRICS (Acquisition)
    # A is 100%, C is 0%
    # ==========================================
    pre_start = reversal_epoch - 20
    pre_end = reversal_epoch
    
    pre_actor_A = np.mean(live_actor['A'][pre_start:pre_end])
    pre_actor_C = np.mean(live_actor['C'][pre_start:pre_end])
    # Healthy: High positive number (prefers A)
    metrics['Pre_Action_Gap_(A-C)'] = pre_actor_A - pre_actor_C 
    
    # Healthy: ~0.0 (knows C is worthless)
    metrics['Pre_Critic_Value_C'] = np.mean(live_critic['C'][pre_start:pre_end])
    
    # ==========================================
    # 2. POST-REVERSAL METRICS (Cognitive Flexibility)
    # A is 0%, C is 100%
    # ==========================================
    post_actor_A = np.mean(live_actor['A'][-20:])
    post_actor_C = np.mean(live_actor['C'][-20:])
    # Healthy: High negative number (prefers C)
    metrics['Post_Action_Gap_(A-C)'] = post_actor_A - post_actor_C
    
    # Healthy: ~0.0 (knows A is now worthless)
    # L-DOPA: Will likely hallucinate value here due to perseveration
    metrics['Post_Critic_Value_A'] = np.mean(live_critic['A'][-20:]) 
    
    # ==========================================
    # 3. REPRESENTATIONAL GEOMETRY (Post-Reversal)
    # ==========================================
    hs, cues, lengths, _, _, _ = extract_hidden_states_rl(model, trial_params, num_trials_per_stim=50, is_reversed=True)
    hs_np = hs.cpu().detach().numpy()
    
    snapshots = {'A': [], 'B': [], 'C': []} 
    for i in range(hs_np.shape[0]):
        trial_len = lengths[i].item()
        anticipation_timestep = max(0, trial_len - 6) 
        snapshots[cues[i]].append(hs_np[i, anticipation_timestep, :])
        
    centroid_A = np.mean(snapshots['A'], axis=0)
    centroid_C = np.mean(snapshots['C'], axis=0)
    metrics['Post_Rep_Distance_A_C'] = np.linalg.norm(centroid_A - centroid_C)
    
    return metrics