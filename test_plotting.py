import torch
import sys
import os

# Add src to python path so we can import modules
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models import ActorCriticRNN
from plotting import plot_predictions
import matplotlib
# Use aggressive non-interactive backend for tests
matplotlib.use('Agg')

model = ActorCriticRNN()
plot_predictions(model)
print("plot_predictions executed successfully")
