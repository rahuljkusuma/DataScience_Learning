import pandas as pd
from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")

data = pd.read_csv('data_p6.csv')
# data.head()

ids = data['Responder_id']
ages = data['Age']

median = data['Age'].median(axis=0)

median_age = median
color = '#fc4f30'

bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plt.hist(ages, bins=bins, edgecolor='black', log=True)

plt.axvline(median_age, color=color, label='Age Median')

plt.legend()

plt.title('Ages of Respondents')
plt.xlabel('Ages')
plt.ylabel('Total Respondents')

plt.tight_layout()





plt.show()
