import matplotlib
matplotlib.use('TKAgg')
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
from util import *
import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import fmin
from scipy.optimize import fminbound
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.animation as animation

X_MIN = -2
X_MAX = 2
Y_MIN = -2
Y_MAX = 2
X_LENGTH = abs(X_MAX-X_MIN)
Y_LENGTH = abs(Y_MAX-Y_MIN)
bounds = [X_MIN, X_MAX, Y_MIN, Y_MAX]
NUM_AGENTS = 10

def inBounds(x, y):
    return (x < X_MAX and x > X_MIN and y < Y_MAX and y > Y_MIN)

class Agent():

    initialDiscount = 0.99
    discount = 0.99
    eta = 0.05
    radius = float(abs(X_MAX-X_MIN))/float(10)

    def __init__(self, ID, walkRadius = radius, x = 0, y = 0, utility = 0):
        self.r = walkRadius
        self.x = x
        self.y = y
        self.ID = ID
        self.utility = utility
        self.score = 0
        self.w = np.zeros((1, Agent.lenWeight))

    lenWeight = NUM_AGENTS-1
    def featureExtractor(self, x, y):
        #return np.array([x, y])
        return np.array([float(euclideanDistance(x, y, agents[i].x, agents[i].y)) for i in range(NUM_AGENTS) if i != self.ID])

    def minf(self, x):
        return -self.prediction(x[0], x[1])

    def bestPredicted(self):
        limits = [(X_MIN, X_MAX), (Y_MIN, Y_MAX)]
        minimum = fmin_l_bfgs_b(self.minf, x0=np.array([self.x, self.y]), approx_grad=True, bounds=limits)
        x = minimum[0][0]
        y = minimum[0][1]
        return [x, y]

    def explore(self, x, y):
        stddev = float(X_LENGTH/4)*Agent.discount
        x = random.gauss(x, stddev)
        y = random.gauss(y, stddev)
        while x < X_MIN or x > X_MAX:
            x = random.gauss(x, stddev)
        while y < Y_MIN or y > Y_MAX:
            y = random.gauss(y, stddev)
        return [x, y]

    def normalize(self, x, y):
        distance = euclideanDistance(self.x, self.y, x, y)
        if distance > self.r:
            angle = np.angle([complex(x-self.x, y-self.y)])
            x = self.r * math.cos(angle) + self.x
            y = self.r * math.sin(angle) + self.y
        return [x, y]

    def updatePosition(self):
        [x, y] = self.bestPredicted()
        [x, y] = self.explore(x, y)
        [x, y] = self.normalize(x, y)

        self.x = x
        self.y = y

    def prediction(self, x, y):
        featureVector = self.featureExtractor(x, y)
        return np.inner(featureVector, self.w)

    def recordUtility(self, utility):
        self.utility = utility
        self.score += utility * Agent.discount
        self.updateWeights()

    def updateWeights(self):
        #self.w = np.matrix([1, 1, 1, 1, 1])
        self.w = self.w - Agent.eta*(self.prediction(self.x, self.y) - self.utility)*self.featureExtractor(self.x, self.y)
        print self.w

def utilityF(agents, agent):
    #return agent.x + agent.y
    return reduce(lambda x, y: x + y, [math.log(float(euclideanDistance(agent.x, agent.y, neighbor.x, neighbor.y))) for neighbor in agents if neighbor != agent])
    #return -euclideanDistance(agent.x, agent.y, 0, 0)

def initializeRandomAgents(numberAgents):
    agents = list()
    for i in range(numberAgents):
        a = Agent(i, x=random.uniform(X_MIN, X_MAX), y=random.uniform(Y_MIN, Y_MAX))
        agents.append(a)
    return agents

def update(agents):
    for agent in agents:
        agent.updatePosition()
    for agent in agents:
        agent.recordUtility(utilityF(agents, agent))

agents = initializeRandomAgents(NUM_AGENTS)

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

particles, = ax.plot([], [], 'bo', ms=6)

rect = plt.Rectangle(bounds[::2], bounds[1] - bounds[0], bounds[3] - bounds[2], ec='none', lw=2, fc='none')
ax.add_patch(rect)

def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    rect.set_edgecolor('none')
    return particles, rect

def animate(i):
    """perform animation step"""
    global box, rect, ax, fig
    update(agents)
    Agent.discount *= Agent.initialDiscount

    ms = int(fig.dpi * 2 * 0.04 * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])

    rect.set_edgecolor('k')
    particles.set_data([agent.x for agent in agents], [agent.y for agent in agents])
    particles.set_markersize(ms)
    return particles, rect

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=300, blit=False, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
