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
from train import train_model
from configs import TRAINING, TASK   



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
        trained_rnn, loss_history, live_parameters = train_model(
            rnn=rnn, 
            dataset=dataset, 
            lr=params["learning_rate"], 
            epochs=epochs, 
            batches_per_epoch=batches_per_epoch, 
            batch_size=batch_size
        )

        # Take an average of the last 3 epochs' loss
        final_loss = np.mean(loss_history[-3:])

        # Save results of this specific run
        run_data = params.copy()
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

def extract_hidden_states(rnn, num_trials_per_stim=100, is_reversed=False):

    #Generates a structured batch and extracts hidden states.
    rnn.eval()
    
    # Create an equal number of A, B, and C trials
    cues = ["A"] * num_trials_per_stim + ["B"] * num_trials_per_stim + ["C"] * num_trials_per_stim
    
    with torch.no_grad():
        inputs, targets, lengths, mask, cues = generate_batch(
        trial_params=TASK,    
        is_reversed=is_reversed,
        cues=cues)
        
        # Run forward pass
        ys, hs = rnn(inputs)
    
    return hs, cues, lengths,inputs, targets

def train_stimulus_decoders(checkpoint="checkpoints/weights_baseline.pth",reversed=False): #To decode hidden states back into stimulus
    
    rnn = ScratchRNN()
    rnn.load_state_dict(torch.load(checkpoint))
    hs, cues, lengths, inputs, targets =extract_hidden_states(rnn,is_reversed=reversed)

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

def train_continuous_decoders(checkpoint="checkpoints/weights_baseline.pth",reversed="Wrong"):
    rnn = ScratchRNN()
    rnn.load_state_dict(torch.load(checkpoint))
    hs, cues, lengths, inputs, targets =extract_hidden_states(rnn,is_reversed=reversed)

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