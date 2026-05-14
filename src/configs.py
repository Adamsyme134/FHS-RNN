SEED = 42

TASK = {
    "stimulus_duration": 10,
    "reward_duration": 1,
    "delay_duration": 20,
    "reward_probs": {
      "A": 1,
      "B": 0.5,
      "C": 0
    }
}

MODEL = {
    "input_size": 3,
    "hidden_size": 64,
    "output_size": 1,
}

TRAINING = {
    "lr": 1e-3,
    "epochs": 100,
    "batches_per_epoch": 200,
    "batch_size": 32
}