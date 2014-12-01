from util import *
import numpy as np

class agent():

    def __init__(self, walkRadius = 5, x = 0, y = 0, utility = 0):
        self.r = walkRadius
        self.x = x
        self.y = y
        self.utility = utility # Previously recorded utility. Note! Utility doesn't account for the discount factor
        self.score = 0 # Sum of utilities over all times

    # Updates the agents own x and y coordinates ARGUMENT HAVE currentPosition
    def updatePosition(self):
        self.x = farthestX(self)
        self.y = farthestY(self)

    def recordUtility(self, utility, discount):
        self.utility = utility
        self.score += utility*discount
        self.updateWeights()

    def updateWeights(self):
        pass
