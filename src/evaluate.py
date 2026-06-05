import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
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
    """
    Executes a grid search over RL-specific hyperparameters and calculates 
    the convergence speed (epochs required to learn the task).
    """
    # 1. Define the search space
    hyperparameter_grid = {
        'learning_rate': [1e-3],# 5e-4, 1e-4], 
        'gamma': [0.5, 0.9, 0.99],          # Temporal discount factor
        'entropy_coef': [0.001, 0.01, 0.05] # Exploration vs Exploitation
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

        # Unique save directory so checkpoints don't overwrite each other
        run_save_dir = f"checkpoints/rl_search_{i}"
        os.makedirs(run_save_dir, exist_ok=True)

        # 2. Train the model
        model, loss_history, live_performance = train_rl_model(
            num_epochs=epochs,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            lr=params["learning_rate"],
            gamma=params["gamma"],
            entropy_coef=params["entropy_coef"],
            save_dir=run_save_dir
        )

        # 3. Calculate Point of Convergence
        # We define convergence as the Expected Value (Critic) reliably hitting 
        # within 15% of the target values (1.0, 0.5, 0.0) for 5 consecutive epochs.
        tolerance = 0.1
        convergence_epoch = -1
        
        total_tracked_epochs = len(live_performance['A'])
        
        for e in range(total_tracked_epochs):
            a_conv = abs(live_performance['A'][e] - 1.0) < tolerance
            b_conv = abs(live_performance['B'][e] - 0.5) < tolerance
            c_conv = abs(live_performance['C'][e] - 0.0) < tolerance
            
            if a_conv and b_conv and c_conv:
                # Check for stability over the next 5 epochs
                if e + 5 < total_tracked_epochs:
                    stable = True
                    for check_e in range(e, e + 5):
                        if not (abs(live_performance['A'][check_e] - 1.0) < tolerance and
                                abs(live_performance['B'][check_e] - 0.5) < tolerance and
                                abs(live_performance['C'][check_e] - 0.0) < tolerance):
                            stable = False
                            break
                    
                    if stable:
                        convergence_epoch = e
                        break

        # 4. Save metrics
        run_data = params.copy()
        run_data['final_loss'] = np.mean(loss_history[-10:]) if loss_history else float('inf')
        run_data['convergence_epoch'] = convergence_epoch if convergence_epoch != -1 else "Did Not Converge"
        run_data['total_timesteps_to_converge'] = (convergence_epoch * batches_per_epoch * batch_size) if convergence_epoch != -1 else "N/A"
        
        all_results.append(run_data)

    # 5. Output Results
    df_results = pd.DataFrame(all_results)
    
    print("\n==========================================")
    print("RL Hyperparameter Search Complete")
    
    # Sort by fastest convergence (ignoring those that didn't converge)
    converged_df = df_results[df_results['convergence_epoch'] != "Did Not Converge"].copy()
    if not converged_df.empty:
        converged_df = converged_df.sort_values(by='convergence_epoch')
        best_run = converged_df.iloc[0]
        print(f"Fastest Convergence: Epoch {best_run['convergence_epoch']}")
        print(f"Best Parameters: LR={best_run['learning_rate']}, Gamma={best_run['gamma']}, Entropy={best_run['entropy_coef']}")
    else:
        print("No combinations successfully converged within the epoch limit.")
        
    return df_results

def extract_hidden_states(model, trial_params, num_trials_per_stim=100, is_reversed=False, noise_stdev=0.1):

    #Generates a structured batch and extracts hidden states.
    model.eval()
    
    # Create an equal number of A, B, and C trials
    cues = ["A"] * num_trials_per_stim + ["B"] * num_trials_per_stim + ["C"] * num_trials_per_stim
    
    with torch.no_grad():
        inputs, targets, lengths, mask, cues = generate_batch(
        trial_params=trial_params,    
        is_reversed=is_reversed,
        cues=cues,
        noise_stdev=noise_stdev)
        
        # Run forward pass
        ys, hs = model(inputs)
    
    return hs, cues, lengths,inputs, targets

def extract_hidden_states_rl(model, num_trials_per_stim=100):
    model.eval()

    stimuli = np.array(
        [0] * num_trials_per_stim +
        [1] * num_trials_per_stim +
        [2] * num_trials_per_stim
    )

    env = RLTask(batch_size=len(stimuli))

    inputs, stimuli, lengths, events = env.get_batch(
        stimuli=stimuli,
        return_events=True
    )

    cue_names = np.array(["A", "B", "C"])
    cues = [cue_names[s] for s in stimuli]

    with torch.no_grad():
        ys, hs = model(inputs)

    return hs, cues, lengths, inputs, events

def train_stimulus_decoders(model, trial_params, reversed=False, noise_stdev=0.1): #To decode hidden states back into stimulus
    
    hs, cues, lengths, inputs, targets =extract_hidden_states(
        model=model, trial_params=trial_params, is_reversed=reversed, noise_stdev=noise_stdev)

    X_all = hs.cpu().numpy()
    y_all_1d = np.array(cues)

    if torch.is_tensor(inputs):
        inputs_all = inputs.cpu().numpy()
    else:
        inputs_all = np.array(inputs)

    batch_size, sequence_length, hidden_dim = X_all.shape

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    first_onset = 1000
    offset_timesteps = []
    per_stimulus_accuracies = {'A': [], 'B': [], 'C': []}
    avg_accuracy = []
    
    print(f"Training decoders on {inputs_all.shape[0]} trials")
    for trial in range(inputs_all.shape[0]):
        stim_mask = np.any(inputs_all[trial, :, :3] == 1, axis=1)  # first 3 input channels

        if np.any(stim_mask):
            onset = np.argmax(stim_mask)              # first timestep with stimulus
            if onset<first_onset:
                first_onset = onset
            offset = len(stim_mask) - np.argmax(stim_mask[::-1]) - 1  # last timestep with stimulus
        
            offset_timesteps.append(offset)

    stimulus_start_timestep = first_onset
    avg_stimulus_end_timestep = np.mean(offset_timesteps) if offset_timesteps else np.nan

    for t in range(sequence_length):
        #slice hidden states for all trials at this specific time point
        X_t = X_all[:, t, :]

        fold_acc_A = []
        fold_acc_B = []
        fold_acc_C = []
        for train_idx, test_idx in cv.split(X_t, y_all_1d):
            X_train, X_test = X_t[train_idx], X_t[test_idx]
            y_train, y_test = y_all_1d[train_idx], y_all_1d[test_idx]
            
            #Train a linear decoder for these hidden states
            decoder = LogisticRegression(max_iter=1000)
            decoder.fit(X_train, y_train)
            
            #Evaluate on held-out trials
            y_pred = decoder.predict(X_test)
           
            #Per-stimulus accuracy
            cm = confusion_matrix(y_test, y_pred, labels=['A', 'B', 'C'])
            class_accs = cm.diagonal() / cm.sum(axis=1)
            fold_acc_A.append(class_accs[0])
            fold_acc_B.append(class_accs[1])
            fold_acc_C.append(class_accs[2])    
        #Average the accuracies for time step T
        per_stimulus_accuracies['A'].append(np.mean(fold_acc_A))
        per_stimulus_accuracies['B'].append(np.mean(fold_acc_B))
        per_stimulus_accuracies['C'].append(np.mean(fold_acc_C))

        avg_accuracy.append(    (
        np.mean(fold_acc_A) +
        np.mean(fold_acc_B) +
        np.mean(fold_acc_C)
        ) / 3)
    
    return per_stimulus_accuracies, avg_accuracy, stimulus_start_timestep, avg_stimulus_end_timestep

def train_continuous_decoders(model, trial_params, reversed=False, noise_stdev=0.1):
    hs, cues, lengths, inputs, targets = extract_hidden_states(
        model=model,
        trial_params=trial_params,
        is_reversed=reversed,
        noise_stdev=noise_stdev
    )

    X_all = hs.cpu().numpy()
    
    # Ensure targets are 2D (batch_size, sequence_length)
    if targets.dim() == 3:
        y_all = targets.squeeze(-1).cpu().numpy()
    else:
        y_all = targets.cpu().numpy()
        
    batch_size, sequence_length, hidden_dim = X_all.shape
    
    # Matrix to store every predicted value for every trial
    predictions = np.zeros((batch_size, sequence_length))
    
    # Use KFold for continuous regression targets
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for t in range(sequence_length):
        X_t = X_all[:, t, :]
        y_t = y_all[:, t]
        
        for train_idx, test_idx in cv.split(X_t, y_t):
            X_train, X_test = X_t[train_idx], X_t[test_idx]
            y_train, y_test = y_t[train_idx], y_t[test_idx]
            
            # Train linear regressor
            decoder = Ridge(alpha=1.0)
            decoder.fit(X_train, y_train)
            
            # Slot the predictions back into the original trial indices
            predictions[test_idx, t] = decoder.predict(X_test)
            
    # Group the predictions by cue and average them
    cues_array = np.array(cues)
    mean_predictions = {}
    
    for cue in ['A', 'B', 'C']:
        # Find all row indices belonging to this specific cue
        cue_indices = np.where(cues_array == cue)[0]
        # Average those rows across the trial dimension (axis=0)
        mean_predictions[cue] = predictions[cue_indices, :].mean(axis=0)
        
    return mean_predictions

