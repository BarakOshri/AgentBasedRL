import matplotlib
matplotlib.use('TKAgg')
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import util
import random
from agent import *
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

X_MIN = -2
X_MAX = 2
Y_MIN = -2
Y_MAX = 2
NUM_AGENTS = 15
bounds = [X_MIN, X_MAX, Y_MIN, Y_MAX]
radius = float(abs(X_MAX-X_MIN))/float(1000)
discount = 0.95

# Given the position of all agents, returns the utility score for a given agent
def utilityF(agents, agent):
    return float(1)/float(5)

def initializeRandomAgents(numberAgents):
    agents = list()
    for i in range(numberAgents):
        a = agent(radius, random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX))
        agents.append(a)
    return agents

def update(agents, discount):
    for agent in agents:
        agent.updatePosition() #[(agent.x, agent.y) for agent in agents]
    for agent in agents:
        agent.recordUtility(utilityF(agents, agent), discount)
        discount = discount*discount

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
    update(agents, discount)

    ms = int(fig.dpi * 2 * 0.04 * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])

    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data([agent.x for agent in agents], [agent.y for agent in agents])
    particles.set_markersize(ms)
    return particles, rect

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=1, blit=False, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
