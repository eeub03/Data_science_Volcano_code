import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

os.chdir("C:/Users/Joe/PycharmProjects/Data_science_Volcano_code")
volcano_data_603314 = pd.read_csv("data/train/513181.csv", )
volcano_data_603314 = volcano_data_603314.to_numpy()

x = list(range(-7500, 8250, 1250))

fig, ax = plt.subplots(1)
ax.hist(volcano_data_603314, bins=10)

# ax.  (np.arange(min(x), max(x)+1, 1250))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which="minor")
plt.title('Volcano ID 513181 (All sensor data)')

plt.show()
