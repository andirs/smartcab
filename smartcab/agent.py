import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'taxi'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # Initialize Q-table that will be updated in the mix
        self.qtable = {} 
        # Counts the steps taken
        self.step_count = 0
        # Counts the steps that were learned
        self.lesson_count = 0 
        # Learning rate
        self.alpha = 0
        # Gamma value
        self.gamma = 0

        self.state_prev = 0
        self.action_prev = 0
        self.reward_prev = 0




    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # Set step_count to 0 for the new trip
        self.step_count = 0
        # For debugging purposes
        print self.qtable

    def max_action(self, state, qtable):
    	return {action:Q for action, Q in qtable[state].items() if Q == max(
    		qtable[state].values())}

    def learn_policy(self, qtable):
    	Q_val = self.qtable[self.state_prev][self.action_prev]
    	Q_val = Q_val + (self.alpha * (self.reward_prev + (self.gamma * max(
    		self.qtable[self.state].values())) - Q_val ))
    	qtable[self.state_prev][self.action_prev] = Q_val

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.lesson_count += 1

        # Update state according to combination out of inputs
        self.state = (
        	('waypoint', self.next_waypoint), 
        	('traffic_light', inputs['light']), 
        	('oncoming', inputs['oncoming']), 
        	('left', inputs['left'])
        	)

        # Select action according to your policy
        # See if state exists in dictionary already
        if self.qtable.has_key(self.state):
        	# If so, get action with best value 
        	# (if multiple exist pick a random from the set of actions)
        	best_action = self.max_action(self.state, self.qtable)
        	action = random.choice(best_action)
        else:
        	self.qtable.update({self.state : {None : 0, 'forward' : 0, 'left' : 0, 'right' : 0}})
        	action = random.choice([None, 'forward', 'left', 'right'])

        #action = self.update_action(self.state)
        #action = random.choice(['forward', 'forward', 'forward', 'forward'])

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Update Q-Table based on reward
        #self.qtable[self.state] = action

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        if self.step_count > 1:
	        Q_val = self.qtable[self.state_prev][self.action_prev]
	    	Q_val = Q_val + (self.alpha * (self.reward_prev + (self.gamma * max(self.qtable[self.state].values())) - Q_val ))
	    	self.qtable[self.state_prev][self.action_prev] = Q_val

    	self.state_prev = self.state
    	self.action_prev = action
    	self.reward_prev = reward
    	self.step_count += 1



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
