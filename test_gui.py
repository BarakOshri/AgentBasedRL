from GUIclass import *
import numpy as np
import time
import matplotlib.pyplot as plt

size = 30
max_area = 10
numIters = 20
numBlobs = 20
pauseTime = .2
screen = GUIclass(0, 0, size, size)
for _ in range(numIters):
    screen.update(np.random.rand(numBlobs)*size, np.random.rand(numBlobs)*size, np.random.rand(numBlobs), np.pi * (max_area * np.random.rand(numBlobs))**2)
    time.sleep(pauseTime)

