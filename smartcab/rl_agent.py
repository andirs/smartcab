import pandas as pd
import random
from collections import Counter
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import math

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

        # Set learning rate alpha (between 0, 1)
        self.alpha = .9
        # Set learning discount gamma (between 0, 1)
        self.gamma = .2
        # Set randomness threshold
        self.epsilon = .95

        # Initialize variables to store previous state, reward amd action
        self.prev_rewards = None
        self.prev_action = None
        self.prev_state = None

        # Crete statistic variables & dicts
        self.trial_count = 0
        self.trial_summary = {}
        for i in range(3):
            self.trial_summary[i] = 0

    def reset(self, destination=None):
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
                take the best action available or a comlete random action
        '''
        q_start = 0
        action_set = [None, 'forward', 'left', 'right']
        # Determine random threshold by float in between (0, 1) with
        # moving towards 0 for large t
        random_threshold = random.random()
        # Sigmoid function for epsilon that moves towards 
        # 1 with high time step t
        self.epsilon = (1 / (1 + math.exp(-t)))
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
                action = random.choice(action_set)
        else:
        	# Initialize new state using q_start
            qtable.update({state : {None : q_start, 'forward' : q_start, 'left' : q_start, 'right' : q_start}})
            action = random.choice(action_set)
        return action

    def learn_policy(self, qtable, state, alpha, gamma, t):
        '''
        Learn policy with set alpha and gamma.
        '''
        if self.trial_count > 0:
            #alpha = alpha/float(self.trial_count)
            q_new = qtable[self.prev_state][self.prev_action]
            q_new = q_new + (alpha * (self.prev_rewards + (gamma * (max(qtable[state].values()))) - q_new))
            qtable[self.prev_state][self.prev_action] = q_new

    def update(self, t):
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
        probabilities = [self.qtable[(self.state, None)], self.qtable[(self.state, 'forward')], self.qtable[(self.state, 'left')], self.qtable[(self.state, 'right')]]

        # Execute action and get reward
        reward = self.env.act(self, action)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # Learn policy based on state, alpha, gamma and t
        self.learn_policy(self.qtable, self.state, self.alpha, self.gamma, t)

        # Remember state, action & reward for learning
        self.prev_state = self.state
        self.prev_action = action
        self.prev_rewards = reward

        # Create stats about reaching the goal
        if (deadline == 0) & (reward < 8):
            self.trial_summary[0] += 1
            print "#" * 20
            print "Trial was unsuccessful."
            print "#" * 20
        else:
            if (reward >= 8):
                self.trial_summary[1] += 1
                print "#" * 20
                print "Trial was successful."
                print "#" * 20

        self.trial_count += 1

    def feature_comparison(self, 
        alpha = [.05, .3, .6, .9, 1], 
        gamma = [.05, .3, .6, .9, 1],
        epsilon = [.05, .3, .6, .9, 1],
        cv = 5,
        ntrials = 100):
        '''
        Do feature comparison based on sets of list.
        @cv: number of crossvalidation steps
        @ntrials: amount of trial runs
        Saves output in static file in smartcab/data/
        '''
        import os
        o_dir = 'smartcab/data'

        if not os.path.exists(o_dir):
            os.makedirs(o_dir)

        alpha_list = alpha
        gamma_list = gamma
        epsilon_list = epsilon

        feature_summary = []
        '''
        feature_summary = pd.DataFrame(
            columns = [
            'alpha', 
            'gamma', 
            'epsilon', 
            'no_success', 
            'success', 
            'steps'])
        '''

        for alpha in alpha_list:
            for gamma in gamma_list:
                for epsilon in epsilon_list:
                    success_summary = pd.DataFrame(
                    	index = [
                    	'no_success', 
                    	'success', 
                    	'steps'])
                    combination_name = [alpha, gamma, epsilon] # if needed
                    e = Environment()
                    a = e.create_agent(LearningAgent)

                    # Set alpha, gamma and epsilon in agent
                    a.alpha = alpha
                    a.gamma = gamma
                    a.epsilon = epsilon

                    # Prepare simulation details
                    e.set_primary_agent(a, enforce_deadline=True)
                    sim = Simulator(e, update_delay=0, display=False)

                    # Set how many validation runs to take
                    validation_no = cv
                    for i in range(validation_no):
                    	# Run simulation n_trials times
                        sim.run(n_trials=ntrials)
                        success_temp = pd.DataFrame.from_dict(a.trial_summary, orient='index')
                        success_temp.index = ['no_success', 'success', 'steps']
                        temp_column_name = 'trial_count_' + str(i+1)
                        success_summary[temp_column_name] = success_temp
                        # Reset counter
                        a.trial_summary[0] = 0
                        a.trial_summary[1] = 0
                        a.trial_summary[2] = 0

                    # Gather aggregate information from summary
                    # And store in temporary attributes for better
                    # readability
                    temp_dat = success_summary.mean(axis=1)[0:]
                    temp_nsuc_mean = temp_dat[0]
                    temp_suc_mean = temp_dat[1]
                    temp_step_mean = temp_dat[2]
                    df = [alpha, gamma, epsilon, temp_nsuc_mean, temp_suc_mean,temp_step_mean]
                    feature_summary.append(df)
        

        summary_comp = pd.DataFrame(feature_summary, 
        	columns = [
        		'alpha',
        		'gamma',
        		'epsilon',
        		'no_success',
        		'success',
        		'steps'])
        summary_comp.to_csv(o_dir + '/rl_agent_feature_comparison.csv', header = True, index = None, sep = ';', mode = 'a')




def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False


    success_summary = pd.DataFrame(index = ['no_success', 'success', 'steps'])
    validation_no = 5

    
    for i in range(validation_no):
        sim.run(n_trials=100)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

        print "Trial Count: ", a.trial_count
        success_temp = pd.DataFrame.from_dict(a.trial_summary, orient='index')
        success_temp.index = ['no_success', 'success', 'steps']
        temp_column_name = 'trial_count_' + str(i+1)
        success_summary[temp_column_name] = success_temp
        a.trial_summary[0] = 0
        a.trial_summary[1] = 0
        a.trial_summary[2] = 0

    print a.probabilities

    for row in a.qtable.iteritems():
        print row
    print success_summary
    success_average = success_summary.mean(axis=1)[0:]
    print "Average: "
    print success_average
    print "Accuracy: ", min(success_average) / max(success_average) 
    print 
    print

    
    #a.feature_comparison(alpha=[0.9, 1], gamma=[0.9, 1], epsilon=[0.9, 1], cv=2,ntrials=100)
    #a.feature_comparison(epsilon=[1])

if __name__ == '__main__':
    run()
