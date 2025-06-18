import time
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

LOC1_EXPECTED_RETURN = 3
LOC1_EXPECTED_REQUEST = 3

LOC2_EXPECTED_RETURN = 2
LOC2_EXPECTED_REQUEST = 4

max_cars = 20
gamma = .9

reward_per_rent = 10
reward_per_move = -2

