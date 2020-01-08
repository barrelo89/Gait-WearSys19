import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import firstHarmonicsAnalysis as fh
from scipy.stats import skew
from scipy.stats import kurtosis
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def CycleDistributionPlot(cycle_data, save_path, facecolor = 'b'):

    data_name = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']
    _, num_col = cycle_data.shape

    for col_idx in range(num_col):

        plt.hist(cycle_data[:, col_idx], bins = np.arange(0, 4, 0.2))#, facecolor = facecolor
        plt.xlabel('Frequency (Hz)', fontsize = 30)
        plt.ylabel('Number of Occurrence', fontsize = 30)
        plt.xticks([0, 1, 2, 3, 4, 5], fontsize = 30)
        plt.yticks([0, 10, 20, 30, 40], fontsize = 30)
        plt.xlim([0, 5])
        plt.ylim([0, 40])
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, data_name[col_idx] + '_cycle_distribution.png'))
        plt.close()

    plt.hist(cycle_data.ravel(), bins = np.arange(0, 5, 0.2))
    plt.xlabel('Frequency (Hz)', fontsize = 30)
    plt.ylabel('Number of Occurrence', fontsize = 30)
    plt.xticks([0, 1, 2, 3, 4, 5], fontsize = 30)
    plt.yticks([0, 50, 100, 150, 200], fontsize = 30)
    plt.xlim([0, 5])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cycle_distribution.png'))
    plt.close()

def PlotOutlier(data, unit_period_length, img_save_path, boundary):

    num_period = int(len(data) / unit_period_length)
    print(num_period)
    #actually not every period has meaningful data which makes a need to filter out some undesired data periods in calculating mean and std
    valid_start_idx, valid_end_idx = int(num_period*0.1), int(num_period*0.4)
    ######

    skew_list = []
    kurtosis_list = []
    std_list = []

    for period_idx in range(num_period):

        skew_list.append(skew(data[period_idx*unit_period_length:(period_idx+1)*unit_period_length]))
        kurtosis_list.append(kurtosis(data[period_idx*unit_period_length:(period_idx+1)*unit_period_length]))
        std_list.append(data[period_idx*unit_period_length:(period_idx+1)*unit_period_length].max())

    sk_uppder, sk_lower = np.array(skew_list)[valid_start_idx: valid_end_idx].mean() + boundary*np.array(skew_list)[valid_start_idx: valid_end_idx].std(), np.array(skew_list)[valid_start_idx: valid_end_idx].mean() - boundary*np.array(skew_list)[valid_start_idx: valid_end_idx].std()
    kur_uppder, kur_lower = np.array(kurtosis_list)[valid_start_idx: valid_end_idx].mean() + boundary*np.array(kurtosis_list)[valid_start_idx: valid_end_idx].std(), np.array(kurtosis_list)[valid_start_idx: valid_end_idx].mean() - boundary*np.array(kurtosis_list)[valid_start_idx: valid_end_idx].std()
    std_uppder, std_lower = np.array(std_list)[valid_start_idx: valid_end_idx].mean() + boundary*np.array(std_list)[valid_start_idx: valid_end_idx].std(), np.array(std_list)[valid_start_idx: valid_end_idx].mean() - boundary*np.array(std_list)[valid_start_idx: valid_end_idx].std()

    fig, ax = plt.subplots()

    #plot the patterns in a given data sequence
    ax.plot(data)
    #mark the std of each data period in scatter plot
    #boundary: 96% trust region
    ax.scatter(range(50, 50 + unit_period_length*num_period, unit_period_length), std_list, color = 'g', label = 'STD')
    ax.hlines(np.array(std_list)[valid_start_idx: valid_end_idx].mean(), xmin = 0, xmax = len(data), colors = 'g')
    ax.hlines(std_lower, xmin = 0, xmax = len(data), colors = 'g', linestyles = 'dashed')
    ax.hlines(std_uppder, xmin = 0, xmax = len(data), colors = 'g', linestyles = 'dashed')
    ax.set_yticks([data.min(), data.max()])
    #ax.set_xlim([0, 3000])

    #kurtosis
    ax2 = ax.twinx()
    ax2.scatter(range(50, 50 + unit_period_length*num_period, unit_period_length), kurtosis_list, color = 'r')
    ax.scatter(50, kurtosis_list[0], color= 'r', label = 'Kurtosis')
    ax2.hlines(np.array(kurtosis_list)[valid_start_idx: valid_end_idx].mean(), xmin = 0, xmax = len(data), colors = 'r')
    ax2.hlines(kur_uppder, xmin = 0, xmax = len(data), colors = 'r', linestyles = 'dashed')
    ax2.hlines(kur_lower, xmin = 0, xmax = len(data), colors = 'r', linestyles = 'dashed')

    #skewness
    ax2.scatter(range(50, 50 + unit_period_length*num_period, unit_period_length), skew_list, color = 'c')
    ax.scatter(50, skew_list[0], color= 'c', label = 'Skewness')
    ax2.hlines(np.array(skew_list)[valid_start_idx: valid_end_idx].mean(), xmin = 0, xmax = len(data), colors = 'c')
    ax2.hlines(sk_uppder, xmin = 0, xmax = len(data), colors = 'c', linestyles = 'dashed')
    ax2.hlines(sk_lower, xmin = 0, xmax = len(data), colors = 'c', linestyles = 'dashed')

    ax.legend()

    valid_img_save_path = os.path.join(img_save_path, 'valid_cycle')

    if not os.path.exists(valid_img_save_path):
        os.makedirs(valid_img_save_path)

    plt.savefig(os.path.join(valid_img_save_path, str(user_idx) + '.png'))
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(skew_list, kurtosis_list, std_list)


    sk_range = [sk_lower, sk_uppder]
    kur_range = [kur_lower, kur_uppder]
    std_range = [std_lower, std_uppder]

    def x_y_edge(x_range, y_range, z_range):

        xx, yy = np.meshgrid(x_range, y_range)

        for value in [0, 1]:
            output = np.array([z_range[value]]*4).reshape(2, 2)
            ax.plot_wireframe(xx, yy, output, color="r")

    def y_z_edge(x_range, y_range, z_range):

        yy, zz = np.meshgrid(y_range, z_range)

        for value in [0, 1]:
            output = np.array([x_range[value]]*4).reshape(2, 2)
            ax.plot_wireframe(output, yy, zz, color="r")

    def x_z_edge(x_range, y_range, z_range):

        xx, zz = np.meshgrid(x_range, z_range)

        for value in [0, 1]:
            output = np.array([y_range[value]]*4).reshape(2, 2)
            ax.plot_wireframe(xx, output, zz, color="r")

    x_y_edge(sk_range, kur_range, std_range)
    y_z_edge(sk_range, kur_range, std_range)
    x_z_edge(sk_range, kur_range, std_range)
    ax.set_xticks(np.linspace(min(skew_list), max(skew_list), 4))
    ax.set_yticks(np.linspace(min(kurtosis_list), max(kurtosis_list), 4))
    ax.set_zticks(np.linspace(min(std_list), max(std_list), 4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Skew', fontsize = 15)
    ax.set_ylabel('Kurtosis', fontsize = 15)
    ax.set_zlabel('STD', fontsize = 15)

    ax.xaxis.set_tick_params(labelsize = 12)
    ax.yaxis.set_tick_params(labelsize = 12)
    ax.zaxis.set_tick_params(labelsize = 12)
    plt.tight_layout()
    plt.savefig(os.path.join(valid_img_save_path, str(user_idx) + '_3D.png'))
    plt.close()

processed_data_base_path = 'processed_data'
img_save_base_path = 'img'
data_save_base_path = 'data'

activities = ['walk']

for activity in activities:

    processed_data_path = os.path.join(processed_data_base_path, activity)
    img_save_path = os.path.join(img_save_base_path, activity)

    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    threshold = 2
    sampling_frequency = 100

    #for reference to column names in the csv file
    data_name = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']#, 'Heart Rate'

    data_save_path = os.path.join(data_save_base_path, activity)
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    if not (os.path.exists(os.path.join(data_save_path, 'input_data.pkl')) and os.path.exists(os.path.join(data_save_path, 'fft.npy'))):

        data_file_list = sorted(os.listdir(processed_data_path))

        cycle_list = []
        input_data_list = []

        for idx, file_name in enumerate(data_file_list):

            data_file_path = os.path.join(processed_data_path, file_name)
            data = pd.read_csv(data_file_path, delimiter = ',').values[1:, 1:-1]
            data = data.astype(float)

            row, col = data.shape

            #peak removal
            data = fh.data_removal_trial(data, threshold)
            print(file_name, data.shape)

            #interpolation -> would be handled in the later part
            #data = fh.filter_data(data)[:, :-1]

            if not os.path.exists(os.path.join(data_save_path, 'input_data.pkl')):
                input_data_list.append(data)

            num_target_column = len(data_name)

            fig, ax = plt.subplots(nrows = num_target_column, ncols = 1)
            user_cycle_list = []

            for col_idx in range(num_target_column):

                col_data = data[:, col_idx]

                n = len(col_data) #length of the signal
                k = np.arange(n)
                Fs = sampling_frequency
                T = n / Fs
                frq = k / T # two sides frequency range
                frq = frq[range(int(n/2))] # one side frequency range

                Y = np.fft.fft(col_data)/n # fft computing and normalization
                Y = Y[range(int(n/2))]

                user_cycle_list.append(frq[np.argmax(abs(Y))])

                ax[col_idx].plot(frq, abs(Y)) # plotting the spectrum
                ax[col_idx].set_xlim([0, 6])

            cycle_list.append(user_cycle_list)
            plt.xlabel('Frequency (Hz)')
            plt.tight_layout()

            fft_img_save_path = os.path.join(img_save_path, 'fft')

            if not os.path.exists(fft_img_save_path):
                os.makedirs(fft_img_save_path)

            user_fft_img_save_path = os.path.join(fft_img_save_path, file_name.split('_')[0] + '.pdf')
            plt.savefig(user_fft_img_save_path)
            plt.close()

        if not os.path.exists(os.path.join(data_save_path, 'input_data.pkl')):
            pickle.dump(input_data_list, open(os.path.join(data_save_path, 'input_data.pkl'), 'wb' ))

        cycle_data = np.array(cycle_list)
        np.save(os.path.join(data_save_path,'fft.npy'), cycle_data)

    else:
        cycle_data = np.load(os.path.join(data_save_path,'fft.npy'))
        input_data = pickle.load(open(os.path.join(data_save_path,'input_data.pkl'), 'rb'))

    #By analyzing the fft of the input data, we can approximate the length of one period in data pattern
    #Here, it is estimated as 100 which is equal to the sampling frequency of the smartwatch used in the experiment
    cycle_hist_save_path = os.path.join(img_save_path, 'cycle_hist')

    if not os.path.exists(cycle_hist_save_path):
        os.makedirs(cycle_hist_save_path)

    #Plot the histogram of detected periods using FFT in the given data sequences
    CycleDistributionPlot(cycle_data, cycle_hist_save_path)


























#end
