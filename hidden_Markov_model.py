#!/usr/bin/env python

from pomegranate import *
import numpy as np

# define first prob
sun = DiscreteDistribution({
    "umbrella" : 0.2,
    "no umbrella" : 0.8
})

# define second prob
rain = DiscreteDistribution({
    "umbrella" : 0.9,
    "no umbrella" : 0.1
})

states = [sun, rain]

# transition model  (Markov model)
# starting probabilites
starts = np.array([0.5, 0.5])

transitions = np.array([[0.8, 0.2],         # tomorrows prediction if today = sun
                       [0.3, 0.7]]        # tomorrows prediction if today = rain
)

#create the model
model = HiddenMarkovModel.from_matrix(
                          transitions, states, starts,
                          state_names=['sun', 'rain']
)

model.bake()

sys.path.append('./AI/hidden_Markov_model.py')
sys.path.insert(0, './AI/hidden_Markov_model.py/')

