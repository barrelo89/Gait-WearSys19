import os
import pickle
import numpy as np
import peakutils as pk
import matplotlib.pyplot as plt

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def Plot(data):

    num_row, num_col = data.shape

    figure, axes = plt.subplots(num_col, sharex = True)

    for col_idx, ax in enumerate(axes):

        ax.plot(data[:, col_idx])
        #ax.plot(data[:2000, col_idx])
        ax.set_title(data_name[col_idx])
        ax.set_yticks([data[100:2000, col_idx].min(), data[100:2000, col_idx].max()])
        ax.axvline(100, color = 'k', linestyle = 'dotted', label = 'Period')
        ax.axvline(200, color = 'k', linestyle = 'dotted')
        ax.axvline(300, color = 'k', linestyle = 'dotted')
        ax.axvline(400, color = 'k', linestyle = 'dotted')
        ax.axvline(500, color = 'k', linestyle = 'dotted')
        ax.axvline(600, color = 'k', linestyle = 'dotted')
        ax.axvline(700, color = 'k', linestyle = 'dotted')
        ax.axvline(800, color = 'k', linestyle = 'dotted')
        ax.axvline(900, color = 'k', linestyle = 'dotted')
        ax.axvline(1000, color = 'k', linestyle = 'dotted')
        ax.axvline(1100, color = 'k', linestyle = 'dotted')
        ax.axvline(1200, color = 'k', linestyle = 'dotted')
        ax.axvline(1300, color = 'k', linestyle = 'dotted')
        ax.axvline(1400, color = 'k', linestyle = 'dotted')
        ax.axvline(1500, color = 'k', linestyle = 'dotted')
        ax.axvline(1600, color = 'k', linestyle = 'dotted')
        ax.axvline(1700, color = 'k', linestyle = 'dotted')
        ax.axvline(1800, color = 'k', linestyle = 'dotted')
        ax.axvline(1900, color = 'k', linestyle = 'dotted')
        #ax.set_xlim([100, 2000])

    plt.tight_layout()

    valid_img_save_path = os.path.join(img_save_path, 'after_valid_detection')

    if not os.path.exists(valid_img_save_path):
        os.makedirs(valid_img_save_path)

    plt.savefig(os.path.join(valid_img_save_path, 'valid_' + str(idx) +  '.png'))
    plt.close()

def PeriodLengthHist(each_period_data):

    cycle_length_list = []

    for user_cycles in each_period_data:

        for cycle in user_cycles:

            cycle_length_list.append(len(cycle))

    plt.hist(cycle_length_list)
    plt.xlabel('Cycle Length', fontsize = 30)
    plt.ylabel('Number of Occurrence', fontsize = 30)
    plt.xticks([0, 50, 100, 150, 200], fontsize = 30)
    plt.yticks([0, 1000, 2000, 3000, 4000], fontsize = 30)
    plt.xlim([0, 200])
    plt.ylim([0, 4000])
    plt.tight_layout()
    plt.show()


sampling_frequency = 100

data_base_path = 'data'
img_save_base_path = 'img'
data_name = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']
activities = ['walk']

for activity in activities:

    data_path = os.path.join(data_base_path, activity)
    img_save_path = os.path.join(img_save_base_path, activity)
    cycle_data = np.load(os.path.join(data_path, 'fft.npy'))[:, -1]

    #filtered and valid data set
    data_list = pickle.load(open(os.path.join(data_path, 'valid_data.pkl'), 'rb'))

    if not os.path.exists(os.path.join(data_path, 'cycles.pkl')):

        period_data_list = []

        for idx, (data, cycle) in enumerate(zip(data_list, cycle_data)):

            unit_period_length = int(sampling_frequency / cycle)

            #extract a valid period of data iteratively
            gyro_z_data = data[:, -1] #we use gyro z data as it shows the most clean data pattern

            num_period = int(len(gyro_z_data) / unit_period_length)

            #tunable parameters: we got this emperically
            guard_range_proportion = 0.4
            guard_size = int(guard_range_proportion*unit_period_length)

            user_period_list = []
            slice_start_idx = 0

            for period_idx in range(num_period):

                print(idx, '%d/%d'%(period_idx+1, num_period))

                target_start_idx = slice_start_idx + unit_period_length - guard_size
                target_end_idx = slice_start_idx + unit_period_length + guard_size

                #find the local minima
                if len(gyro_z_data[target_start_idx : target_end_idx]):
                    local_min_idx = np.argmin(gyro_z_data[target_start_idx : target_end_idx])

                    #define the end idx for slicing a valid period
                    slice_end_idx = target_start_idx + local_min_idx

                    plt.plot(gyro_z_data[slice_start_idx:slice_end_idx])
                    plt.ylim(gyro_z_data.min(), gyro_z_data.max())
                    cycle_img_save_path = os.path.join(img_save_path, 'cycle')
                    if not os.path.exists(cycle_img_save_path):
                        os.makedirs(cycle_img_save_path)

                    plt.savefig(os.path.join(cycle_img_save_path, str(idx) + '_' + str(period_idx) + '.pdf'))
                    plt.close()

                    user_period_list.append(data[slice_start_idx:slice_end_idx, :])

                    slice_start_idx = slice_end_idx

            period_data_list.append(user_period_list)

        pickle.dump(period_data_list, open(os.path.join(data_path, 'cycles.pkl'), 'wb' ))

    else:
        period_data_list = pickle.load(open(os.path.join(data_path, 'cycles.pkl'), 'rb' ))

    PeriodLengthHist(period_data_list)
    






























































#end
