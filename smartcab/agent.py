# Import necessary packages
from collections import Counter
from environment import Agent, Environment
from operator import add
from planner import RoutePlanner
from simulator import Simulator

import math
import numpy as np
import os
import pandas as pd
import pickle
import random

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):

        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'taxi'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # Additional variables:
        # Initialize qtable to store state and action
        self.qtable = {}

        # Counts the steps our algorithm has learned
        self.learn_count = 0

        # Time Series information
        self.cycles = 0 # How many cycles did the program run through
        self.time_count_pos = [0] * 100
        self.time_count_neg = [0] * 100

        self.deadline_start = 0
        self.deadline_start_col = []
        self.deadline_end_col = []

        # Set learning rate alpha (between 0, 1)
        self.alpha = .9
        # Set learning discount gamma (between 0, 1)
        self.gamma = .2
        # Set randomness threshold
        self.epsilon = .95
        self.q_start = {None : 10, 'forward': 10, 'left': 10, 'right': 10}
        #self.q_start = {None : 0, 'forward': 0, 'left': 0, 'right': 0}
        self.q_start_count = {None : 0, 'forward': 0, 'left': 0, 'right': 0}

        # Initialize variables to store previous state, reward amd action
        self.prev_rewards = None
        self.prev_action = None
        self.prev_state = None

        # Crete statistic variables & dicts
        self.trial_count = 0
        self.trial_summary = {}
        for i in range(5):
            self.trial_summary[i] = 0

    def deadline_stats(self, start, deadline):
        if (start):
            self.deadline_start_col.append(deadline)
        elif not start:
            self.deadline_end_col.append(deadline)

    def success_stats(self, suc):
        if (suc):
            self.trial_summary[1] += 1
            self.time_count_pos[self.cycles] += 1
        else:
            self.trial_summary[0] += 1
            self.time_count_neg[self.cycles] += 1

    def reset(self, destination=None):
        self.cycles += 1
        self.planner.route_to(destination)
        # Set trial_count to 0 again
        self.trial_summary[2] = self.trial_count
        self.trial_count = 0

    def best_action(self, qtable, state, t):
        '''
        Decide best action for a certain state:
            If state hasn't been seen before:
                Initialize state with a default-value.
            If state has been seen before:
                Decide with a certain randomness whether to
                take the best action available or a comlete random action.
        '''
        action_set = [None, 'forward', 'left', 'right']
        # Determine random threshold by float in between (0, 1) with
        # moving towards 0 for large t
        random_threshold = random.random()
        if self.epsilon == 9:
            # Sigmoid function for epsilon that moves towards 1
            self.epsilon = (1 / (1 + math.exp(-t)))
        elif self.epsilon == 99:
            # 'Slower' sigmoid function
            self.epsilon = (1 / (1 + math.exp((-t/2))))
        elif self.epsilon == 999:
            # Even 'slower' sigmoid function
            self.epsilon = (1 / (1 + math.exp((-t/3))))
        # Check if state exists in qtable already
        if qtable.has_key(state):
            # Check if random threshold is smaller or equal epsilon
            # to account for randomness factor
            if random_threshold < self.epsilon:
                # Determine maximum of Q value(s) for given state
                q_max = max(qtable[state].values())
                action_set = {action:q_new for action, q_new in qtable[state].items() if q_new == q_max}
                action = random.choice(action_set.keys())
            else:
            	# Occasionally perform random action
                # to explore world especially in the beginning
                action = random.choice(action_set)
        else:
        	# Initialize new state using q_start
            qtable.update({state : {None : self.q_start[None], 'forward' : self.q_start['forward'], 'left' : self.q_start['left'], 'right' : self.q_start['right']}})
            action = random.choice(action_set)
            self.q_start_count[action] += 1
        return action

    def learn_policy(self, qtable, state, alpha, gamma, t):
        '''
        Learn policy based on alpha, gamma 
        and previous rewards.
        '''
        if alpha > 8:
            # Make sigmoid start at almost 0
            t_alpha = t-5
        if alpha == 9:
            # Sigmoid function that moves towards zero for high t
            alpha = (1 / (1+math.exp(t_alpha)))
        elif alpha == 99:
            # 'Slower' sigmoid function
            alpha = (1 / (1+math.exp(t_alpha/2)))
        elif alpha == 999:
            # Even 'slower' sigmoid function
            alpha = (1 / (1+math.exp(t_alpha/3)))

        if gamma > 8:
            # Make sigmoid start at almost 0
            t_gamma = t-5
        if gamma == 9:
            # Sigmoid function for gamma
            gamma = (1 / (1+math.exp(-t_gamma)))
        elif gamma == 99:
            # Slow sigmoid function for gamma
            gamma = (1 / (1+math.exp(-t_gamma/2)))
        elif gamma == 999:
            # Slow sigmoid function for gamma
            gamma = (1 / (1+math.exp(-t_gamma/3)))

        if self.trial_count > 0:
            q_new = qtable[self.prev_state][self.prev_action]
            q_new = (1-alpha) * q_new + (alpha * (self.prev_rewards + (gamma * (max(qtable[state].values())))))
            qtable[self.prev_state][self.prev_action] = q_new

    def update(self, t):
        '''
        Combine determination of action, calculation of reward
        and learning process of agent.
        Afterwards, check if agent has reached goal and report.
        '''
        # Gather inputs for current state
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.learn_count += 1  #count steps in total run

        # Set current state based on inputs and next waypoint
        self.state = (
            ("waypoint",self.next_waypoint),
            ("light",inputs['light']),
            ("traffic_oncoming", inputs['oncoming']),
            ("traffic_left",inputs['left']),
            ("traffic_right",inputs['right'])
            )

        # Find best action to take given current state
        action = self.best_action(self.qtable, self.state, t)

        # Execute action and get reward
        reward = self.env.act(self, action)
        #if (self.q_start_count[action] == 1):
        #    self.q_start[action] = reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # Learn policy based on state, alpha, gamma and t
        self.learn_policy(
            self.qtable, self.state, self.alpha, self.gamma, t)

        # Remember state, action & reward for learning
        self.prev_state = self.state
        self.prev_action = action
        self.prev_rewards = reward

        self.trial_count += 1

    def feature_comparison(self, 
        output,
        alpha = [.05, .3, .6, .9, 1, 9, 99], 
        gamma = [.05, .3, .6, .9, 1, 9, 99],
        epsilon = [.05, .3, .6, .9, 1, 9, 99],
        cv = 5,
        ntrials = 100):
        '''
        Feature comparison based on sets of list.
        @alpha: list of alpha values
        @gamma: list of gamma values
        @epsilon: list of epsilon values
        @cv: number of validation steps
        @ntrials: amount of trial runs
        Saves output in static file in 'data/'
        '''
        # Set folder for output
        o_dir = 'data'
        if not os.path.exists(o_dir):
            os.makedirs(o_dir)

        # Time Series Setup
        ts_store = {}

        # For naming conventions
        alpha_list = alpha
        gamma_list = gamma
        epsilon_list = epsilon

        # Create container for the whole summary
        feature_summary = []

        # Create combinations for each of the possible settings
        for alpha in alpha_list:
            for gamma in gamma_list:
                for epsilon in epsilon_list:
                    success_summary = pd.DataFrame(
                    	index = [
                    	'no_success', 
                    	'success', 
                    	'steps',
                        'deadline_start', 
                        'deadline_finish', 
                        'percentage'])

                    e = Environment()
                    a = e.create_agent(LearningAgent)

                    # Set alpha, gamma and epsilon in agent
                    a.alpha = alpha
                    a.gamma = gamma
                    a.epsilon = epsilon

                    # Set up time slot summary for all trials
                    sum_counts_neg = [0] * 100
                    sum_counts_pos = [0] * 100
                    temp_store = {}


                    # Prepare simulation details
                    e.set_primary_agent(a, enforce_deadline=True)
                    sim = Simulator(e, update_delay=0, display=False)

                    # Set how many validation runs to take
                    validation_no = cv
                    for i in range(validation_no):
                    	# Run simulation n_trials times
                        sim.run(n_trials=ntrials)
                        
                        # Deal with temp data and store
                        a.trial_summary[3] = np.mean(a.deadline_start_col)
                        a.trial_summary[4] = np.mean(a.deadline_end_col)
                        a.trial_summary[5] = (a.trial_summary[4] / a.trial_summary[3])*100
                        success_temp = pd.DataFrame.from_dict(a.trial_summary, orient='index')
                        success_temp.index = ['no_success', 'success', 'steps', 'deadline_start', 'deadline_finish', 'percentage']
                        temp_column_name = 'trial_count_' + str(i+1)
                        success_summary[temp_column_name] = success_temp
                        # Reset counter and temp data
                        a.trial_summary[0] = 0
                        a.trial_summary[1] = 0
                        a.trial_summary[2] = 0
                        a.trial_summary[3] = 0
                        a.trial_summary[4] = 0
                        a.trial_summary[5] = 0
                        a.deadline_start_col = []
                        a.deadline_end_col = []

                        # Sum up counts per time slot for negative
                        # and positive outcomes
                        sum_counts_neg = map(add, sum_counts_neg, a.time_count_neg)
                        sum_counts_pos = map(add, sum_counts_pos, a.time_count_pos)
                    
                    # prep dict for each slot by calculating the average for
                    # each time step
                    temp_store['neg'] = map(
                        lambda x: x/float(cv), sum_counts_neg)
                    temp_store['pos'] = map(
                        lambda x: x/float(cv), sum_counts_pos)
                    
                    # Store each slot in one big dict that we can use
                    ts_store[(alpha, gamma, epsilon)] = temp_store

                    # Gather aggregate information from summary
                    # And store in temporary attributes for better
                    # readability
                    temp_dat = success_summary.mean(axis=1)[0:]
                    temp_nsuc_mean = temp_dat[0]
                    temp_suc_mean = temp_dat[1]
                    temp_step_mean = temp_dat[2]
                    temp_dls_mean = temp_dat[3]
                    temp_dle_mean = temp_dat[4]
                    temp_perc_mean = temp_dat[5]
                    df = [alpha, gamma, epsilon, temp_nsuc_mean, temp_suc_mean,temp_step_mean, temp_dls_mean, temp_dle_mean, temp_perc_mean]
                    feature_summary.append(df)
        
        # Turn summary into data frame
        summary_comp = pd.DataFrame(feature_summary, 
        	columns = [
        		'alpha',
        		'gamma',
        		'epsilon',
        		'no_success',
        		'success',
        		'steps',
                'deadline_start',
                'deadline_end',
                'percentage'])
        # Write data frame to disk
        summary_comp.to_csv(o_dir + '/' + output, header = True, index = None, sep = ';', mode = 'a')

        pickle.dump(ts_store, open(o_dir + '/ts_' + output.split('.')[0] + '.p', 'wb'))


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    ###
    ### Run the program without stats
    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


    ###
    ### Stats for analysis
    ### Perform feature comparison with a variety of settings
    ### to find best choice of alpha, gamma and epsilon.

    # Q_start = 0; no first reward action value
    #a.feature_comparison(output='feature_comparison_qinit0_rinit0.csv')

    # Same setup but more granular parameters
    #a.feature_comparison(output='feature_comparison_granular.csv', alpha=[.05, .1, .2, 99], gamma=[.6, .7, .8, 99], epsilon=[.9, .95, .99])

    # Q_start = 10; no first reward action value
    #a.feature_comparison(output='feature_comparison_qinit10_rinit0.csv', alpha=[99], gamma=[99], epsilon=[.99])

    # Q_start = 0; with first reward action value
    #a.feature_comparison(output='feature_comparison_qinit0_rinit1.csv', alpha=[99], gamma=[99], epsilon=[.99])

if __name__ == '__main__':
    run()
