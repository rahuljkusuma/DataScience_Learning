from matplotlib import pyplot as plt

import numpy as np
from collections import Counter
import csv

"""
# Developers ages & salaries
dev_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]

# Python developers ages & salaries
py_dev_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

# JavaScript developers salaries
js_dev_y = [37810, 43515, 46823, 49293, 53437,
            56373, 62375, 66674, 68745, 68746, 74583]

dummy_y = [32010, 40015, 41023, 43393, 50437,
            51073, 52075, 53074, 55245, 60046, 65383]

#As the ages of both developers are same, lets create a list of ages for plotting both salaries
ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] 


plt.style.use('ggplot')

x_indexes = np.arange(len(ages_x))
width = 0.20

plt.bar(x_indexes-(width), dev_y, width=width, color='#444444', linestyle='--', label='All Devs')
plt.bar(x_indexes, py_dev_y, width=width, color='#008fd5', label='Python')
plt.bar(x_indexes+(width), js_dev_y, width=width, color="#e5ae38", label="JavaScript")

# plt.bar(x_indexes+(width*1.5), dummy_y, width=width, color="#800080", label="Dummy")


plt.legend()

plt.xticks(ticks=x_indexes, labels=ages_x)

plt.title('Median Salary (USD) by Age')
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')

plt.tight_layout()

plt.show()

"""

with open('data_p2.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    language_counter = Counter()
    
    for row in csv_reader:
        language_counter.update(row['LanguagesWorkedWith'].split(';'))
        


languages = [i[0] for i in language_counter.most_common(15)]
popularity = [i[1] for i in language_counter.most_common(15)]


plt.style.use("ggplot")


plt.barh(languages[::-1], popularity[::-1]) #For horizontal bar charts use "barh"

plt.title("Most Popular Languages")
# plt.ylabel("Programming Languages")
plt.xlabel("Number Of Users")

# plt.legend()

plt.tight_layout()

plt.show()















































