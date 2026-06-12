SEED = 42

TASK = {
    "stimulus_duration": 10,
    "reward_duration": 5,
    "delay_duration": 15, #20,
    "baseline_duration": 0,
    "reward_probs": {
      "A": 1,
      "B": 0.5,
      "C": 0
    },
    "noise_stdev":0.1
}

MODEL = {
    "input_size": 3,
    "hidden_size": 128,
    "output_size": 1,
}


TRAINING = {
    "lr": 1e-3,
    "epochs": 500,#500,
    "batches_per_epoch": 50,
    "batch_size": 32,
    "reversal_epoch":250,#300\
    "gamma":0.99,
    "phenotypes": {
    "Healthy_Baseline": {"alpha_plus": 1.0, "alpha_minus": 1.0},
    "PD_Untreated": {"alpha_plus": 0.4, "alpha_minus": 1.5},
    "PD_LDOPA": {"alpha_plus": 1.5, "alpha_minus": 0.1}
}
}