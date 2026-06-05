#Will generate the tasks for the model training/ eval

#Time structure: ITI -> Stimulus -> Delay -> Reward
#At each timestep, input should be x_t = [stim_A, stim_B, stim_C, reward]
#At t=0            [0,0,0,0]
#During stimulus A [1,0,0,0]
#During reward     [0,0,0,1]

#desired licks = 1 for A, 0.5 for B, 0 for C
#ITI random between 3-8 timesteps
import random
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from scipy.ndimage import gaussian_filter1d

def parse_duration(duration_config):
    if isinstance(duration_config, (list, tuple)):
        return np.random.randint(duration_config[0], duration_config[1] + 1)
    return int(duration_config)

def add_guassian_noise(inputs, std_dev=0.1):
    noise = np.random.normal(loc=0.0, scale=std_dev, size=inputs.shape)
    noisy_inputs = inputs + noise

    #Clip to keep inputs within 0-1
    noisy_inputs = np.clip(noisy_inputs, 0.0, 1.0)

    return noisy_inputs

def generate_trial(cue_choice, reward_choice, trial_params, SIGMA=1.2, is_reversed = False, noise=True, noise_stdev=0.1):
    noise_stdev= 0.1
    stimulus_duration = parse_duration(trial_params["stimulus_duration"])
    delay_duration = parse_duration(trial_params["delay_duration"])
    reward_duration = parse_duration(trial_params["reward_duration"])
    if not is_reversed:
        reward_probs = trial_params["reward_probs"]
    else:
        # Create a temporary reversed dictionary. B stays the same
        reward_probs = {'A': 0.0, 'B': trial_params["reward_probs"]['B'], 'C': 1.0}

    cue = random.choice(["A", "B", "C"]) if cue_choice == "RANDOM" else cue_choice
    cue_int = {'A': 0, 'B': 1, 'C': 2}[cue]
    expected_value = reward_probs[cue]

    rewarded = int(np.random.rand() < reward_probs[cue]) if reward_choice =="RANDOM" else int(reward_choice) #decides if reqard will be given based on reqard probability
    iti_length = np.random.randint(3,9) #ITI interval between 3-8ts (upper bound exclusive in 3-9)
    T = iti_length + stimulus_duration + delay_duration + reward_duration #Total trial length

    inputs = np.zeros((T,4)) #Creates array of 0 with T rows and 4 columns ([0,0,0,1] etc as above)
    targets = np.zeros((T,1)) #creates output array

    inputs[iti_length:(iti_length+stimulus_duration) ,cue_int] = 1 #fills the cue period with the correct stimulus channel
    reward_period = slice(
        iti_length + stimulus_duration + delay_duration
    )
    inputs[(T-reward_duration):T, 3] = rewarded #fills reward period

    #Decide desired licking behaviour
    anticipation_period = slice(
        iti_length + stimulus_duration,
        iti_length + stimulus_duration + delay_duration 
    )

    #Smooth out just the anticipation period
    plateau_target = np.zeros_like(targets)
    # This starts at the anticipation period and goes all the way to the end of the trial
    plateau_target[iti_length + stimulus_duration:] = expected_value
    ramp_smoothed = gaussian_filter1d(plateau_target, sigma=SIGMA, axis=0)
    targets[:T-1, 0] = ramp_smoothed[:T-1, 0]

    targets[T-1,0] = rewarded #gives the reward (if randomly selected) 

    if noise:
        inputs = add_guassian_noise(inputs, std_dev=noise_stdev)
    
    return inputs,targets

def generate_batch(*,
    trial_params,
    trial_counts={"A": 11, "B": 10, "C": 11}, 
    cues=None,
    rewards=["RANDOM"],
    is_reversed=False,
    SIGMA=1.5,
    noise=True,
    noise_stdev=0.1):
    if cues is not None:
        cues = cues
        batch_size = len(cues)
    else:
        cues = [] #Create list of cues based on specified counts
        for cue_type, count in trial_counts.items():
            cues.extend([cue_type] * count)

        random.shuffle(cues) #Shuffle the cues
        batch_size = len(cues)

    batch_inputs = []
    batch_targets =[]
    lengths = []

    for t in range(batch_size):
        #Allows for specific plotting trials
        cue_choice = cues[t]
        
        reward_choice = rewards[t] if (len(rewards) > 1 or rewards[0] != "RANDOM") else "RANDOM"
        inputs, targets = generate_trial(
            cue_choice=cue_choice,
            reward_choice=reward_choice,
            trial_params=trial_params,
            SIGMA=SIGMA,
            is_reversed=is_reversed,
            noise=noise,
            noise_stdev=noise_stdev
        )
        inputs = torch.tensor(inputs, dtype = torch.float32)
        targets = torch.tensor(targets, dtype = torch.float32)

        batch_inputs.append(inputs)
        batch_targets.append(targets)
        lengths.append(inputs.shape[0])

    lengths = torch.tensor(lengths)

    #pads to the longest sequence in the batch, to ensure all trials are the same length
    padded_inputs = pad_sequence(
        batch_inputs,
        batch_first = True,
        padding_value = 0.0
    )

    padded_targets = pad_sequence(
        batch_targets,
        batch_first = True,
        padding_value = 0.0        
    )

    max_length = padded_inputs.shape[1]

    # Creates a mask to show which values in each trial are real (1) and which are padding (0)
    mask = (
        torch.arange(max_length)[None, :]
        < lengths[:, None]
    ).int()

    return padded_inputs, padded_targets, lengths, mask, cues

def generate_full_dataset(epochs, trial_params, trial_counts, batches_per_epoch, reversal_epoch, SIGMA=1.5):

    print(f"Generating full dataset for {epochs} epochs...")
    
    dataset= {
        "train": [],
        "val": [], #Add later
        "test": [], #Add later
        "probe": None
    }

    for e in range(epochs):
        is_reversed = e >= reversal_epoch
        epoch_batches = []
        for _ in range(batches_per_epoch):
            batch = generate_batch(
                trial_params=trial_params,
                trial_counts=trial_counts,
                is_reversed=is_reversed, 
                SIGMA=SIGMA
            )
            epoch_batches.append(batch)
        dataset["train"].append(epoch_batches) #each index is an epoch

    
    with torch.no_grad():
        probe_inputs, padded_targets, probe_lengths, mask, cues_probe = generate_batch(
            trial_params=trial_params,
            trial_counts={"A": 20, "B": 20, "C": 20},
            is_reversed=False, 
            SIGMA=SIGMA
        )    
        dataset["probe"] = (probe_inputs, padded_targets, probe_lengths, mask, cues_probe)
    
    print("Dataset generation completed.")
    return dataset

class RLTask:
    def __init__(
        self,
        batch_size=32,
        stimulus_duration=10,
        delay_duration=5,
        reward_duration=5
    ):
        self.batch_size = batch_size
        
        # Reward probabilities for Stimuli A, B, C (indices 0, 1, 2)
        self.reward_probs = {0: 1.0, 1: 0.5, 2: 0.0} #Keep as this? 0.8/0.2?
        # Reward and penalty values
        self.reward_value = 1.0
        self.lick_cost = -0.01

        self.stimulus_duration = stimulus_duration
        self.delay_duration = delay_duration
        self.reward_duration = reward_duration

        self.iti_lengths = np.random.randint(3, 9, size=self.batch_size)
        self.stim_starts = self.iti_lengths
        self.stim_ends = self.stim_starts + self.stimulus_duration
        self.reward_starts = self.stim_ends
        self.reward_ends = self.reward_starts + self.reward_duration
        self.seq_lens = self.reward_ends
        self.max_seq_len = int(np.max(self.seq_lens))

    
    def get_batch(self, stimuli=None, return_events=False):

        self.iti_lengths = np.random.randint(3, 9, size=self.batch_size) 
        self.stim_starts = self.iti_lengths
        self.stim_ends = self.stim_starts + self.stimulus_duration
        self.reward_starts = self.stim_ends
        self.reward_ends = self.reward_starts + self.reward_duration
        self.seq_lens = self.reward_ends
        self.max_seq_len = int(np.max(self.seq_lens))

        #Randomly choose stimuli
        if stimuli is None:
            stimuli = np.random.choice([0, 1, 2], size=self.batch_size)

        # Create the zeroed input tensor using the max sequence length
        inputs = torch.zeros((self.batch_size, self.max_seq_len, 3))
        
        #Populate the stimulus into the input tensor trial-by-trial
        for b in range(self.batch_size):
            stim_class = stimuli[b]
            start = self.stim_starts[b]
            end = self.stim_ends[b]
            
            # Apply 1.0 only during this trial's specific stimulus window
            inputs[b, start:end, stim_class] = 1.0

        if return_events:
            events = []
            for b in range(self.batch_size):
                events.append({
                    "stim_on": int(self.stim_starts[b]),
                    "stim_last": int(self.stim_ends[b] - 1),
                    "reward_on": int(self.reward_starts[b]),
                    "reward_last": int(self.reward_ends[b] - 1),
                    "trial_end": int(self.seq_lens[b] - 1),
                })
            return inputs, stimuli, torch.tensor(self.seq_lens), events

        return inputs, stimuli

    def evaluate_sequence(self, stimuli, actions):
        #Evaluates model's action at every time step
        #Currently, action-1 is Go, action=0 is no go
        rewards = torch.zeros((self.batch_size, self.max_seq_len))

        for b in range(self.batch_size):
            stim = stimuli[b]
            prob = self.reward_probs[stim]
            
            # Determine if this specific trial actually yields reward
            trial_rewarded = np.random.binomial(1, prob) #Along binomial -- Normal?
            reward_harvested = False

            #Timings for this specific trial
            actual_len = self.seq_lens[b]
            trial_stim_end = self.stim_ends[b]

            for t in range(actual_len):
                
                if actions[b, t] == 1: # The model chose to lick
                    if t >= trial_stim_end: #In outcome window
                        # Check if the lick is valid for a reward (is there a reward, has it already been licked)
                        if trial_rewarded and not reward_harvested:
                            rewards[b, t] = self.reward_value
                            reward_harvested = True # Consume the reward
                       
                        elif not trial_rewarded:
                            rewards[b, t] = self.lick_cost # Penalize licking when no reward is present
                        else:
                            # No penalty for extra licking in outcome window
                            rewards[b, t] = 0 #
                    else:
                        # Penalize licking prematurely
                        rewards[b, t] = self.lick_cost
                            
        return rewards

