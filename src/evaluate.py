import itertools
import numpy as np
from task import *
from models import *
from train import *
from configs import TRAINING
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import ScratchRNN
import pandas as pd

def run_hyperparameter_search():


    #Hyperparameter ranges to try out
    hyperparameter_grid = {
        'learning_rate':[5e-3,1e-3,1e-4,1e-5], #[1e-3, 5e-4],
        'hidden_size':[128], #[128,256],
        'target_sigma':[1.2] #[1.2,1.5]
    }
    #get constant parameters
    epochs = TRAINING["epochs"]
    batches_per_epoch = TRAINING["batches_per_epoch"]
    batch_size = TRAINING["batch_size"]

    #generates all combinations of hyperparameters
    keys, values = zip(*hyperparameter_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total experiments to run: {len(experiments)}")

    best_loss = float('inf')
    best_params = None

    #Search loop
    all_results = []
    for i, params in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)} ---")
        print(f"Parameters: {params}")

        SIGMA = params["target_sigma"]

        #Train the model with selected parameters
        rnn = ScratchRNN(input_size=4, hidden_size=params['hidden_size'], output_size=1)
        trained_rnn, loss_history, live_parameters = train_model(rnn, params["learning_rate"], epochs, batches_per_epoch, batch_size,SIGMA)

        #Take an average of the last 3 epochs' loss
        final_loss = np.mean(loss_history[-3:])

        #Save results of this specific run
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

