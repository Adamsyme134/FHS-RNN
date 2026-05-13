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
reward_probs = TASK["reward_probs"]
stimulus_duration = TASK["stimulus_duration"]
delay_duration = TASK["delay_duration"]
reward_duration = TASK["reward_duration"]



cue = random.choice(["A", "B", "C"])
cue_int = {'A': 0, 'B': 1, 'C': 2}[cue]
expected_value = reward_probs[cue]
rewarded = int(np.random.rand() < reward_probs[cue]) #decides if reqard will be given based on reqard probability
iti_length = np.random.randint(3,9) #ITI interval between 3-8ts (upper bound exclusive in 3-9)
T = iti_length + stimulus_duration + delay_duration + reward_duration #Total trial length

inputs = np.zeros((T,4)) #Creates array of 0 with T rows and 4 columns ([0,0,0,1] etc as above)
targets = np.zeros((T,1)) #creates output array


inputs[iti_length:(iti_length+stimulus_duration) ,cue_int] = 1 #fills the cue period with the correct stimulus channel
reward_period = slice(
    iti_length + stimulus_duration + delay_duration
)
inputs[(T-reward_duration):T, 3] = rewarded#rewarded #fills reward period

#Decide desired licking behaviour
anticipation_period = slice(
    iti_length + stimulus_duration,
    iti_length + stimulus_duration + delay_duration 
)
targets[anticipation_period] = expected_value #simplest version, trains model to predict the lick rate exactly the whole period
targets[T-1,0] = 3 #rewarded #gives the reward (if randomly selected) 

for i in range(T):
    print(inputs[i],"|",targets[i])
print("iti_length: ",iti_length)
print(stimulus_duration)
print("prob of reward: ",expected_value,"| rewarded: ",rewarded)