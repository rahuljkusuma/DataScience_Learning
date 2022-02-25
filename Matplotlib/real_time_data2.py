import random
import csv
import time
from itertools import count

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

x_value = 0
total_1 = 1000
total_2 = 1000

fieldnames = ["x_value","total_1","total_2"]

with open('data_p9_1.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()


"""
Below Code is to plot continuous random numbers by adding/substracting it from 1000 (for total_1 & total_2)

"""

while True:
    
    try:
    
        ask = input("Do you want to write another row?(Yes/No): ")

        if ask[0].lower() not in ['y','n']:
            print("Please enter valid response.")
            True

        elif ask[0].lower()=='y':

            with open('data_p9_1.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                info = {
                    'x_value':x_value,
                    'total_1':total_1,
                    'total_2':total_2
                }
                csv_writer.writerow(info)
                
                x_value += 1
                total_1 = total_1 + random.randint(-6, 8)
                total_2 = total_2 + random.randint(-5, 8)


        elif ask[0].lower()=="n":
            print("Thank you")
            break
            
    except:
        print("Please enter valid response.")
        continue
        
        
        
"""
Plotting
"""
def animate(i):
    
    df = pd.read_csv("data_p9_1.csv")
    x = df.x_value
    y1 = df.total_1
    y2 = df.total_2
    
    plt.cla()
    plt.plot(x, y1, label='Channel 1')
    plt.plot(x, y2, label='Channel 2')
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    
ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()
        
        
        
        
        
        
        
        
        
        

        
        
