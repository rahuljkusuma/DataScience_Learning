from matplotlib import pyplot as plt

plt.style.use("ggplot")


data = pd.read_csv('data_p2.csv')
# data.head()

ids = data['Responder_id']
lang_response = data['LanguagesWorkedWith']

lang_counter = Counter()
for i in lang_response:
    lang_counter.update(i.split(';'))

lang_counter = lang_counter.most_common(5)
# lang_counter

langs = [i[0] for i in lang_counter]
# langs

popularity = [i[1] for i in lang_counter]
# popularity



slices = popularity
labels = langs
#If we have less than or 5 items to plot then we can use pie chart.

explode = [0, 0, 0, 0.1, 0]

plt.title("My Awesome Pie Chart")
plt.tight_layout()

#  colors=colors,

plt.pie(slices, labels=labels, explode=explode, shadow=True, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor':'black'})


plt.show()



