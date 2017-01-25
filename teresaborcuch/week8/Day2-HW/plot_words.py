#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd

# make dataframe to hold words
df = pd.read_csv('output.txt', sep = ' ')
df.columns = ['word', 'count']

df.sort_values('count', inplace = True, ascending = False)
df.reset_index(drop = True)

# make plot
bar_heights = df['count'].values[:20]
labels = df['word'].values[:20]
plt.bar(range(1, 21), bar_heights, 1)
plt.xticks(range(1,21), labels, rotation = 40)
plt.savefig('word_plot.png')
