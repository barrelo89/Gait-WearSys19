import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def cos_sim(array_a, array_b):
    return np.dot(array_a, array_b)/(np.linalg.norm(array_a)*np.linalg.norm(array_b))

def cycle_filter(cycle_data):

    filtered_data = []

    for user_idx, user_cycles in enumerate(cycle_data):

        feature_matrix = []

        for cycle in user_cycles:

            gyro_z_cycle = cycle[:, -1]

            feature_matrix.append([gyro_z_cycle.max(), gyro_z_cycle.std(), skew(gyro_z_cycle), kurtosis(gyro_z_cycle)])


        feature_matrix = np.array(feature_matrix)
        clf = LocalOutlierFactor(n_neighbors = 20, novelty = False, contamination = 0.05)
        prediction = clf.fit_predict(feature_matrix) #1 for inliers -1 for outliers

        outliers = user_cycles[prediction == -1]
        inliers = user_cycles[prediction == 1]
        filtered_data.append(inliers)

    return filtered_data

def data_abstract(cycle_data):

    avg_data_list = []

    for user_idx, user_data in enumerate(cycle_data):

        avg_user_data = user_data.mean(axis = 0)
        avg_data_list.append(avg_user_data)

    return avg_data_list

data_base_path = 'data'
img_save_base_path = 'img'
activities = ['walk']

for activity in activities:

    data_path = os.path.join(data_base_path, activity)
    img_save_path = os.path.join(img_save_base_path, activity)

    cycle_data = pickle.load(open(os.path.join(data_path, 'interpolation.pkl'), 'rb'))
    filtered_img_save_path = os.path.join(img_save_path, 'filtered')

    if not os.path.exists(filtered_img_save_path):
        os.makedirs(filtered_img_save_path)

    avg_data_list = data_abstract(cycle_data)

    filtered_user_cycle_data = []

    for user_idx, (user_avg, user_cycle_data) in enumerate(zip(avg_data_list, cycle_data)):

        print(user_idx, len(user_cycle_data))
        user_cos_sim_list = []

        for cycle_idx, cycle in enumerate(user_cycle_data):
            user_cos_sim_list.append(cos_sim(user_avg[:, -1], cycle[:, -1]))

        user_cos_sim_list = np.array(user_cos_sim_list).reshape(-1, 1)

        clf = KMeans(n_clusters = 2)
        labels = clf.fit_predict(user_cos_sim_list)

        if len(user_cos_sim_list[labels == 0]) > len(user_cos_sim_list[labels == 1]):
        #if user_cos_sim_list[labels == 0].mean() > user_cos_sim_list[labels == 1].mean():
            filtered_user_cycle_data.append(user_cycle_data[labels == 0, :, :])
            print(user_cycle_data[labels == 0, :, :].shape)
            normal_idx = 0
            anomal_idx = 1

        elif len(user_cos_sim_list[labels == 0]) < len(user_cos_sim_list[labels == 1]):
        #elif user_cos_sim_list[labels == 0].mean() < user_cos_sim_list[labels == 1].mean():
            filtered_user_cycle_data.append(user_cycle_data[labels == 1, :, :])
            print(user_cycle_data[labels == 1, :, :].shape)
            normal_idx = 1
            anomal_idx = 0

        #print(len(user_cos_sim_list[labels == 0]), len(user_cos_sim_list[labels == 1]))
        plt.scatter(range(len(user_cos_sim_list[labels == normal_idx])), user_cos_sim_list[labels == normal_idx], c = 'b', label = 'Inliers')
        plt.scatter(range(len(user_cos_sim_list[labels == anomal_idx])), user_cos_sim_list[labels == anomal_idx], c = 'r', label = 'Outliers')
        plt.hlines(0, xmin = 0, xmax = max(len(user_cos_sim_list[labels == normal_idx]), len(user_cos_sim_list[labels == anomal_idx])), linestyles = 'dashdot')
        plt.legend(loc = 'best', fontsize = 20)
        plt.xticks(np.linspace(0, max(len(user_cos_sim_list[labels == normal_idx]), len(user_cos_sim_list[labels == anomal_idx])), 5, dtype = np.int32), fontsize = 20)
        plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize = 20)
        plt.xlabel('Cycle Idx', fontsize = 20)
        plt.ylabel('Cosine Similarity', fontsize = 20)
        plt.ylim([-1, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(filtered_img_save_path, str(user_idx) + '.png'))
        plt.close()
        #plt.show()

    pickle.dump(filtered_user_cycle_data, open(os.path.join(data_path, 'filtered_interpolation.pkl'), 'wb'))































#end
