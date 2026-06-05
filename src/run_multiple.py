import os
import torch
import time
import random
import numpy as np
from configs import TASK, TRAINING
from task import generate_full_dataset
from models import ScratchRNN, RLModelWrapper, ActorCriticRNN
from train import train_model, train_rl_model
from plotting import plot_trajectories_with_error, aggregate_continuous_decoding

def set_global_seed(seed):
    #Ensures each run is uniquely randomized but perfectly reproducible later.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_variance_experiment(num_runs=5, model_type="sl"):
    print(f"Starting {num_runs} runs for {model_type.upper()} model...")
    start_time = time.time()
    baseline_checkpoints = []
    
    for i in range(num_runs):
        seed = i + 42 # Arbitrary starting seed
        set_global_seed(seed)
        
        run_save_dir = f"checkpoints/run_seed_{seed}"
        os.makedirs(run_save_dir, exist_ok=True)
        print(f"\n======================================")
        print(f" STARTING RUN {i+1}/{num_runs} (Seed: {seed})")
        print(f"======================================")
        
        if model_type == "sl":
            dataset = generate_full_dataset(
                epochs=TRAINING["epochs"],
                trial_params=TASK,             
                trial_counts={"A": 10, "B": 10, "C": 12}, 
                batches_per_epoch=TRAINING["batches_per_epoch"],
                reversal_epoch=TRAINING["reversal_epoch"],
            )
            rnn = ScratchRNN()
            train_model(
                rnn, dataset, TRAINING["lr"], TRAINING["epochs"], 
                TRAINING["batches_per_epoch"], TRAINING["batch_size"], 
                probe=True, save_dir=run_save_dir
            )
            model_to_evaluate = rnn
            
        elif model_type == "rl":
            rnn, _, _ = train_rl_model(
                num_epochs=TRAINING["epochs"], batch_size=TRAINING["batch_size"], 
                lr=TRAINING["lr"], save_dir=run_save_dir
            )
            model_to_evaluate = RLModelWrapper(rnn)

        # Track the baseline checkpoint from this specific run
        baseline_checkpoints.append(os.path.join(run_save_dir, "weights_baseline.pth"))

    print("\n--- All Runs Complete. Generating SE Decoding Plots ---")
    
    # Run the multi-checkpoint aggregation using the new plotting logic
    summary_stats = aggregate_continuous_decoding(
        model=model_to_evaluate, # An empty architecture to load weights into
        trial_params=TASK,
        checkpoints=baseline_checkpoints,
        reversed=False
    )
    
    plot_trajectories_with_error(
        summary_stats=summary_stats,
        reversed=False
        # Note: To include the vertical stimulus lines, you will need to also pass 
        # avg_stimulus_start and avg_stimulus_end here by returning them from a stimulus decoder.
    )

    # Calculate and print the total time elapsed
    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n✅ Experiment finished in {mins}m {secs}s")
if __name__ == "__main__":
    choice = input("Run variance experiment for SL or RL? (sl/rl): ").lower()
    run_variance_experiment(num_runs=5, model_type=choice)