SEED = 42

TASK = {
    "stimulus_duration": 10,
    "reward_duration": 5,
    "delay_duration": 0, #20,
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
    "lr": 3e-4,
    "epochs": 400,
    "batches_per_epoch": 50,
    "batch_size": 32,
    "reversal_epoch":200
}