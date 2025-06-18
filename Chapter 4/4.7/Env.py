import time
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

class Environment():
    def __init__(self, cars, loc1_requests, loc2_requests,\
                loc1_returns, loc2_returns, action_space, sale,\
                move):
        self.cars = cars
        self.loc1_requests = loc1_requests
        self.loc2_requests = loc2_requests
        self.loc1_returns = loc1_returns
        self.loc2_returns = loc2_returns
        