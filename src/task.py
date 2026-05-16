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
from configs import TASK
import torch
from torch.nn.utils.rnn import pad_sequence
from scipy.ndimage import gaussian_filter1d

def generate_trial(cue_choice, reward_choice, SIGMA=1.2):
    reward_probs = TASK["reward_probs"]
    stimulus_duration = TASK["stimulus_duration"]
    delay_duration = TASK["delay_duration"]
    reward_duration = TASK["reward_duration"]

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

    #Apply guassian smoothing to the target line
    
    return inputs,targets


def generate_batch(batch_size=4,cues=["RANDOM"],rewards=["RANDOM"],SIGMA=1.5):
    batch_inputs = []
    batch_targets =[]
    lengths = []

    for t in range(batch_size):
        #Allows for specific plotting trials
        cue_choice = cues[t] if cues[0] != "RANDOM" else "RANDOM"
        
        reward_choice = rewards[t] if rewards[0] != "RANDOM" else "RANDOM"
        inputs, targets = generate_trial(cue_choice,reward_choice,SIGMA)
        
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

    return padded_inputs, padded_targets, lengths, mask


