import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

target_period_length = 200

data_base_path = 'data'
img_save_base_path = 'img'
activities = ['walk']

for activity in activities:

    data_path = os.path.join(data_base_path, activity)

    #load each sliced period
    each_period_data = pickle.load(open(os.path.join(data_path, 'cycles.pkl'), 'rb' ))

    new_cycle_data = []

    for user_cycles in each_period_data:

        new_user_cycles = []

        for cycle in user_cycles:

            cycle_length, num_type = cycle.shape

            new_cycle_group = []

            for col_idx in range(num_type):

                x = range(cycle_length)
                f = interp1d(x, cycle[:, col_idx], kind = 'linear')

                new_x = np.linspace(0, cycle_length-1, num = target_period_length)
                new_cycle = f(new_x)

                #visualization
                plt.plot(cycle[:, col_idx], color = 'r', label = 'Original')
                plt.plot(new_cycle, color = 'b', label = 'Interpolated')
                plt.legend(loc = 'best', fontsize = 15)
                plt.xticks([0, 50, 100, 150, 200], fontsize = 15)
                plt.yticks(np.linspace(cycle[:, col_idx].min(), cycle[:, col_idx].max(), 4, dtype = np.int32), fontsize = 15)
                plt.xlabel('Sample Idx', fontsize = 15)
                plt.tight_layout()
                plt.show()

                new_cycle_group.append(new_cycle.reshape(-1, 1))

            new_cycle_group = np.concatenate(new_cycle_group, axis = 1)
            new_user_cycles.append(new_cycle_group)

        new_user_cycles = np.array(new_user_cycles)
        new_cycle_data.append(new_user_cycles)

    #after interpolation, save the quantized periods
    pickle.dump(new_cycle_data, open(os.path.join(data_path, 'interpolation.pkl'), 'wb'))




















































#end
