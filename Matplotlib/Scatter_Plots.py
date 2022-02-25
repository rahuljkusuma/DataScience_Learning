import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('seaborn')

df = pd.read_csv('data_p7.csv')

view_count = df.view_count
likes = df.likes
ratio = df.ratio

plt.scatter(view_count, likes, c=ratio, cmap='summer', s=100,
            edgecolors='black', linewidths=1, alpha=0.75)

cbar = plt.colorbar()
cbar.set_label('Like/Dislike Ratio')

plt.xscale('log')
plt.yscale('log')

plt.title('Trending Youtube Videos')
plt.xlabel('View Count')
plt.ylabel('Total Likes')

plt.tight_layout()

plt.show()