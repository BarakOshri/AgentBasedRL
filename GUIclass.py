import numpy as np
import matplotlib.pyplot as plt

class GUIclass:
    # min x coord, min y coord, max x coord, max y coord, size of screen in inches
    def __init__(self, xmin, ymin, xmax, ymax, size=8):
        self.xrange = [xmin, xmax]
        self.yrange = [ymin, ymax]
        plt.figure(num=0, figsize=(size,size), facecolor='w')
        plt.show(block=False)

    def set_bounds(self):
        plt.scatter(self.xrange, self.yrange, s=[0, 0])

    # takes a vector of x coords, y coords, colors, and areas, and draws them
    def update(self, x, y):
        plt.clf()
        self.set_bounds()
        plt.scatter(x, y, s=10, alpha=0.5)
        plt.draw()


