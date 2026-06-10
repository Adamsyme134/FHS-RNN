import os
import shutil
from pathlib import Path
import torch
import time
import random
import numpy as np
import scipy.stats as stats
from configs import TASK, TRAINING
from task import generate_full_dataset
from models import ScratchRNN, RLModelWrapper, ActorCriticRNN
from train import train_model, train_rl_model
from plotting import plot_trajectories_with_error, aggregate_continuous_decoding, plot_all_graphs, plot_phenotype_overlay_timeline, plot_variance_batch_timeline

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
            rnn, _, _, _ = train_rl_model(
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


# 1. Define the clinical phenotypes based on D1/D2 pathway dynamics
PHENOTYPES = {
    "Healthy_Baseline": {"alpha_plus": 1.0, "alpha_minus": 1.0},
    "PD_Untreated": {"alpha_plus": 0.4, "alpha_minus": 1.5},
    "PD_LDOPA": {"alpha_plus": 1.5, "alpha_minus": 0.1}
}
def run_phenotype_overlays():
    """Trains 1 seed of each phenotype and overlays them."""
    print("\nGenerating Overlays for Critic and Actor...")
    run_dir = Path("./results/phenotype_overlays")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    virtual_epoch_length = 1000
    total_time = TRAINING["epochs"] * virtual_epoch_length
    
    critic_data = {}
    actor_data = {}
    
    for name, params in PHENOTYPES.items():
        print(f"Training {name} for overlay...")
        _, _, live_critic, live_actor = train_rl_model(
            total_timesteps=total_time,
            batch_size=TRAINING["batch_size"],
            lr=TRAINING["lr"],
            alpha_plus=params["alpha_plus"],
            alpha_minus=params["alpha_minus"],
            save_dir=f"checkpoints/{name}",
            gamma=TRAINING["gamma"]
        )
        critic_data[name] = live_critic
        actor_data[name] = live_actor
        
    plot_phenotype_overlay_timeline(critic_data, TRAINING["reversal_epoch"], metric_name="Critic Value", run_dir=run_dir)
    plot_phenotype_overlay_timeline(actor_data, TRAINING["reversal_epoch"], metric_name="Actor Lick Probability", run_dir=run_dir)
    print("Overlay plotting complete.\n")


def run_variance_timeline_experiment(num_runs=5, condition_name="Healthy_Baseline"):
    """Runs multiple seeds for a specific phenotype and calculates standard error."""
    print(f"\nStarting {num_runs} runs for variance tracking ({condition_name})...")
    params = PHENOTYPES[condition_name]
    run_dir = Path(f"./results/variance_{condition_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    virtual_epoch_length = 1000
    total_time = TRAINING["epochs"] * virtual_epoch_length
    
    all_critic_runs = {'A': [], 'B': [], 'C': []}
    all_actor_runs = {'A': [], 'B': [], 'C': []}
    
    for i in range(num_runs):
        seed = i + 42
        # Assuming you have set_global_seed from your original file
        # set_global_seed(seed) 
        
        print(f"  Run {i+1}/{num_runs}...")
        _, _, live_critic, live_actor = train_rl_model(
            total_timesteps=total_time,
            batch_size=TRAINING["batch_size"],
            lr=TRAINING["lr"],
            alpha_plus=params["alpha_plus"],
            alpha_minus=params["alpha_minus"],
            save_dir=f"checkpoints/var_{condition_name}_{seed}",
            gamma=TRAINING["gamma"]
        )
        
        for cue in ['A', 'B', 'C']:
            all_critic_runs[cue].append(live_critic[cue])
            all_actor_runs[cue].append(live_actor[cue])
            
    # Calculate Mean and SE
    def calc_stats(data_dict):
        stats_out = {}
        for cue in ['A', 'B', 'C']:
            stacked = np.vstack(data_dict[cue])
            stats_out[cue] = {
                'mean': np.mean(stacked, axis=0),
                'se': stats.sem(stacked, axis=0)
            }
        return stats_out

    critic_stats = calc_stats(all_critic_runs)
    actor_stats = calc_stats(all_actor_runs)
    
    plot_variance_batch_timeline(critic_stats, TRAINING["reversal_epoch"], metric_name="Critic Value", run_dir=run_dir)
    plot_variance_batch_timeline(actor_stats, TRAINING["reversal_epoch"], metric_name="Actor Lick Probability", run_dir=run_dir)
    print(f"Variance plotting complete. Saved to {run_dir}\n")

if __name__ == "__main__":
    # 1. Overlay
    #run_phenotype_overlays()
    
    # 2. SE / Variance tracking (Example: test L-DOPA with 5 seeds)
    run_variance_timeline_experiment(num_runs=3, condition_name="PD_Untreated")
# if __name__ == "__main__":
    #choice = input("Run variance experiment for SL or RL? (sl/rl): ").lower()
    #run_variance_experiment(num_runs=5, model_type=choice)