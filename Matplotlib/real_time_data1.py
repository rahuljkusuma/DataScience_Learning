import random
import csv
import time
from itertools import count

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

"""
Below Code is to plot continuous random numbers from 0 to 20

"""

x_vals = []
y_vals = []

index = count()

def animate(i):
    x_vals.append(next(index))
    y_vals.append(random.randint(0, 20))
    
    plt.cla()
    plt.plot(x_vals, y_vals)
    
    
ani = FuncAnimation(plt.gcf(), animate, interval=250)

plt.tight_layout()
plt.show()



