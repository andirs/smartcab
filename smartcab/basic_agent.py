import pandas as pd
import random
from collections import Counter
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'taxi'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        # Simple statistical counter variables
        self.trial_count = 0
        self.trial_summary = {}
        for i in range(2):
            self.trial_summary[i] = 0
        self.cycles = 0
        self.time_count_pos = {}
        self.time_count_neg = {}
        self.deadline_start_col = [0] * 100
        self.deadline_end_col = [0] * 100


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
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trial_count = 0
        self.cycles += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        
        # TODO: Select action according to your policy
        action = random.choice([None, 'forward', 'left', 'right'])

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    success_summary = pd.DataFrame(index = ['no_success', 'success'])
    validation_no = 10

    for i in range(validation_no):
        sim.run(n_trials=100)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
        print "Trial Count: ", a.trial_count
        success_temp = pd.DataFrame.from_dict(a.trial_summary, orient='index')
        success_temp.index = ['no_success', 'success']
        temp_column_name = 'trial_count_' + str(i+1)
        success_summary[temp_column_name] = success_temp
        a.trial_summary[0] = 0
        a.trial_summary[1] = 0
    
    print success_summary
    success_average = success_summary.mean(axis=1)
    print "Average: "
    print success_average
    print "Percentage: ", success_average[0:][1] / success_average[0:][0]

    import os
    filename = 'smartcab/data/basic_agent_trials.csv'
    filename = os.path.join(filename)
    success_summary.to_csv(filename)
    print success_summary

if __name__ == '__main__':
    run()
