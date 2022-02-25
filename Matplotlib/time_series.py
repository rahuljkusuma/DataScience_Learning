import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates

plt.style.use('seaborn')

df = pd.read_csv('data_p8.csv')

df.Date = pd.to_datetime(df.Date)
df.sort_values('Date', inplace=True)

price_date = df.Date
price_close = df.Close

plt.plot_date(price_date, price_close, linestyle='solid')

plt.gcf().autofmt_xdate() #Date alignment

plt.title('Bitcoin Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')

plt.tight_layout()

plt.show()