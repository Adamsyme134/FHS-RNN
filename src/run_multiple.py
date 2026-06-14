import os
import shutil
from pathlib import Path
import torch
import time
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import scipy.stats as stats
from evaluate import train_continuous_decoders, calculate_phenotype_metrics
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
PHENOTYPES = TRAINING["phenotypes"]
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

def run_phenotype_scorecard(epochs=400, num_seeds=5):
    """Trains all phenotypes across multiple seeds and generates a robust quantitative breakage scorecard."""
    print("\n" + "="*80)
    print(f" STARTING ROBUST PHENOTYPE SWEEP ({num_seeds} Seeds/Condition)")
    print("="*80)
    
    scorecard = []
    virtual_epoch_length = 1000
    total_time = epochs * virtual_epoch_length
    
    for condition_name, params in TRAINING["phenotypes"].items():
        print(f"\n--- Simulating {condition_name} ---")
        
        # Accumulators for this condition's metrics
        condition_metrics = {
        }
        
        for seed in range(42, 42 + num_seeds):
            print(f"  > Training Seed {seed}...")
            set_global_seed(seed)
            
            # Train the model specific to this phenotype and seed
            rnn, _, live_critic, live_actor = train_rl_model(
                total_timesteps=total_time,
                bptt_horizon=TRAINING.get("bptt_horizon", 60),
                batch_size=TRAINING["batch_size"],
                lr=TRAINING["lr"],
                alpha_plus=params["alpha_plus"],
                alpha_minus=params["alpha_minus"],
                save_dir=f"checkpoints/scorecard_{condition_name}_seed_{seed}",
                gamma=TRAINING["gamma"]
            )
            
            wrapped_model = RLModelWrapper(rnn)
            
            # Calculate metrics for this specific seed
            seed_metrics = calculate_phenotype_metrics(wrapped_model, live_critic, live_actor, TASK)
            
            if not condition_metrics:
                condition_metrics = {key: [] for key in seed_metrics.keys()}
                
            # Store the metrics
            for key in condition_metrics.keys():
                condition_metrics[key].append(seed_metrics[key])
                
        # Calculate Mean and Standard Error for the entire condition
        avg_metrics = {'Phenotype': condition_name}
        for key, values in condition_metrics.items():
            avg_metrics[f"{key}_Mean"] = np.mean(values)
            avg_metrics[f"{key}_SE"] = stats.sem(values)
            
        scorecard.append(avg_metrics)
        
    # Format and print the results
    df = pd.DataFrame(scorecard)
    df = df.set_index('Phenotype')
    
    # Reorder columns so Means and SEs are next to each other
    ordered_cols = []
    for key in condition_metrics.keys():
        ordered_cols.extend([f"{key}_Mean", f"{key}_SE"])
    df = df[ordered_cols]
    
    print("\n\n" + "="*100)
    print(f" FINAL ROBUST PHENOTYPE SCORECARD (n={num_seeds} per condition)")
    print("="*100)
    print(df.to_markdown(floatfmt=".4f"))
    print("="*100 + "\n")
    
    # Save to CSV
    run_dir = Path("./results/scorecards")
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "robust_phenotype_breakage_metrics.csv")
    return df

def test_decoder_architectures(checkpoint_path, model_type="sl"):
    # Load your trained network
    if model_type == "sl":
        model = ScratchRNN(input_size=4, hidden_size=256, output_size=1)
    # else setup RL model...
    elif model_type == "rl":
        # RL uses 3 inputs (cues only) and a default hidden size of 64
        base_rl = ActorCriticRNN(input_size=3, hidden_size=64, num_actions=2)
        model = RLModelWrapper(base_rl)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    decoders_to_test = ["ols", "ridge", "lasso", "rf"]
    results = {}
    
    for dec in decoders_to_test:
        print(f"Decoding with {dec.upper()}...")
        mean_preds, T_pre, T_stim, T_delay, T_rew, T_post = train_continuous_decoders(
            model, TASK, reversed=False, model_type=model_type, decoder_type=dec
        )
        results[dec] = mean_preds
        
    # Plotting comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True, sharex=True)
    axes = axes.flatten()
    colors = {'A': 'crimson', 'B': 'forestgreen', 'C': 'royalblue'}
    
    for i, dec in enumerate(decoders_to_test):
        ax = axes[i]
        for cue, trajectory in results[dec].items():
            ax.plot(trajectory, linewidth=2, label=f'Cue {cue}', color=colors.get(cue, 'black'))
        
        ax.set_title(f"Decoder: {dec.upper()}")
        ax.axvline(x=T_pre, color='k', linestyle=':', alpha=0.5)
        ax.axvline(x=T_pre + T_stim, color='k', linestyle=':', alpha=0.5)
        ax.axvline(x=T_pre + T_stim + T_delay, color='k', linestyle=':', alpha=0.5)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.show()


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
    
    plot_variance_batch_timeline(critic_stats, TRAINING["reversal_epoch"], metric_name="Critic Value", run_dir=run_dir, live_performance=live_critic)
    plot_variance_batch_timeline(actor_stats, TRAINING["reversal_epoch"], metric_name="Actor Lick Probability", run_dir=run_dir,live_performance=live_critic)
    print(f"Variance plotting complete. Saved to {run_dir}\n")

if __name__ == "__main__":
    #test_decoder_architectures("checkpoints/weights_baseline.pth", model_type="rl")
    # 1. Overlay
    #run_phenotype_overlays()
    
    # 2. SE / Variance tracking (Example: test L-DOPA with 5 seeds)
    #run_variance_timeline_experiment(num_runs=5, condition_name="PD_Untreated")

    #3. Scorecards
    run_phenotype_scorecard(epochs=400, num_seeds=5)