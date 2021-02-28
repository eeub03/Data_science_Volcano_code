import pandas as pd
import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
os.chdir("C:/Users/Joe/PycharmProjects/Data_science_Volcano_code")
volcano_train_labels = pd.read_csv("data/train.csv")

volcano_data_603314 = pd.read_csv("data/train/603314.csv")
volcano_data_603314 = volcano_data_603314.to_numpy()

combination_array = [1,2,3,4,5,6,7,8,9,10]
combined_array = []
combined_array = [",".join(map(str, comb)) for comb in combinations(combination_array, 2)]

print(combined_array)
print()

index = 0

plt.suptitle('Volcano segment ID 513181')
for i in range(len(combined_array)):
        combination = combined_array[index]
        plt.figure()
        plt.xlim([-6500, 6500])
        plt.ylim([-6500, 6500])
        plt.title('Sensor {} vs Sensor {}'.format(combination[0], combination[2:]))
        plt.xlabel("Sensor {}".format(combination[0]))
        plt.ylabel("Sensor {}".format(combination[2]))

        x = volcano_data_603314[:, int(combination[0])]
        y = volcano_data_603314[:, int(combination[2])]
        plt.grid()
        plt.plot(x,y,marker='*',markersize=2, linestyle='none')
        plt.savefig('figure_model/figure_2_{}vs{}.png'.format(combination[0], combination[2:]))
        plt.close()
        index += 1






