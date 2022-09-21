# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:45:16 2022

@author: Titan
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# Bring some raw data.
frequencies = [0.000972222,0.000609568,0.000601852,0.000902778,0.005594136]

freq_series = pd.Series(frequencies)

y_labels = ['10-50 dB \n (26 dB)','15-50 dB\n (26 dB)','20-50 dB\n (26 dB)','25-30 dB\n (26 dB)','30-35 dB\n (26 dB)']

# Plot the figure.
plt.figure(figsize=(10, 8))
ax = freq_series.plot(kind='barh')
ax.set_title('Model comparison of BER when the FEC limit is overcome',fontsize = 16)
ax.set_xlabel('BER below FEC limit (10e-3)',fontsize = 15)
ax.set_ylabel('Model',fontsize = 15)
ax.set_yticklabels(y_labels)
#ax.set_xlim(20, 30) # expand xlim to make labels easier to read

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 10
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.6f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
                                    # positive and negative values.

plt.savefig("image.png",format='jpeg', dpi=500,bbox_inches='tight')
