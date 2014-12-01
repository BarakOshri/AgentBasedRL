from math import *

# Returns the euclidean distance between agents a and b
def euclideanDistance(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def farthestX(agent):
    if agent.x + agent.r < 2: return agent.x + agent.r
    else: return 2

def farthestY(agent):
    if agent.y + agent.r < 2: return agent.y + agent.r
    else: return 2

