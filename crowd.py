# Thanks to Jake Vanderplas for starter code with Matplotlib and animation
import matplotlib
matplotlib.use('TKAgg')
import numpy as np
from math import *
from util import *
import random
from GUIclass import *
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

maximize = True
X_MIN = -2.5
X_MAX = 2.5
X_LENGTH = abs(X_MIN-X_MAX)
Y_MIN = -2.5
Y_MAX = 2.5
Y_LENGTH = abs(Y_MIN-Y_MAX)
bounds = [X_MIN, X_MAX, Y_MIN, Y_MAX]
discount = 0.95
NUM_AGENTS = 10
eta = 0.1

class agent():

    def __init__(self, ID, walkRadius = X_LENGTH/20, x = 0, y = 0, utility = 0):
        self.r = walkRadius
        self.x = x
        self.y = y
        self.ID = ID
        self.utility = utility # Previously recorded utility. Note! Utility doesn't account for the discount factor
        self.score = 0 # Sum of utilities over all times
        self.w = np.zeros((1, NUM_AGENTS))
        self.phi = np.zeros((1, NUM_AGENTS))
        self.prediction = -1

    # Updates the agents own x and y coordinates ARGUMENT HAVE currentPosition
    def updatePosition(self):
        [self.x, self.y] = scipy.optimize.fmin(lambda x: -self.getPrediction[self.x, self.y] if euclideanDistance(self.x, self.y, x[0], x[1]) < self.r else 0)

    def getPrediction(self, x, y):
        self.phi = np.array([euclideanDistance(x, y, agent[i].x, agent[i].y) for i in range(NUM_AGENTS)])
        return np.inner(self.phi, self.w)

    def recordUtility(self, utility, discount):
        self.utility = utility
        self.score += utility*discount
        self.updateWeights()

    def updateWeights(self):
        self.w = self.w - eta*(self.prediction - self.utility)*self.phi

# Given the position of all agents, returns the utility score for a given agent
def utilityF(agents, agent):
    return reduce(lambda x, y: x+y, [euclideanDistance(agent, neighbor) for neighbor in agents if neighbor is not agent])

def initializeRandomAgents(nAgents):
    agents = list()
    for i in range(nAgents):
        a = agent(i, x = random.uniform(X_MIN, X_MAX), y = random.uniform(Y_MIN, Y_MAX))
        agents.append(a)
    return agents

def update(agents):
    print "hello"
    for agent in agents:
        agent.updatePosition()
    for agent in agents:
        agent.recordUtility(utilityF(agent), discount)
    discount = discount*discount


# ANIMATION
############################################################################################################################

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect = 'equal', autoscale_on = False, xlim = (X_MIN-1, X_MAX+1), ylim = (Y_MIN-1, Y_MAX+1))
particles, = ax.plot([], [], 'bo', ms=6)
rect = plt.Rectangle(bounds[::2], bounds[1] - bounds[0], bounds[3] - bounds[2], ec='none', lw=2, fc='none')
ax.add_patch(rect)

# Init function for the graphical interface
def init():
    global rect
    particles.set_data([], [])
    rect.set_edgecolor('None')
    return particles, rect

# Perform one time step in the graphics
def animate(i):
    print "hello"
    global rect, ax, fig
    update(agents)

    rect.set_edgecolor('k')
    ms = int(fig.dpi * 2 * fig.get_figwidth() / np.diff(ax.get_xbound())[0])
    particles.set_data([agent.x for agent in agents], [agent.y for agent in agents])
    particles.set_markersize(ms)
    return particles, rect

agents = initializeRandomAgents(NUM_AGENTS)
animation.FuncAnimation(fig, animate, frames=600, interval=1000, blit = True, init_func = init)
plt.show()
