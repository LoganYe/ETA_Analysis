#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

start_lng = []
start_lat = []
end_lng = []
end_lat = []
start_timestamp = []
duration = []

all_lng = []
all_lat = []

with open('../data/train.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    cnt = 0
    for row in spamreader:
        if 0 == cnt:
            cnt += 1
            continue
        #if cnt < 10000:
        #    continue
        #elif cnt >= 20000:
        #    break
        if cnt > 100000:
            break
        cnt += 1
        #print(row)
        
        start_lng.append(float(row[1]))
        start_lat.append(float(row[2]))
        end_lng.append(float(row[3]))
        end_lat.append(float(row[4]))
        start_timestamp.append(int(row[5]))
        duration.append(int(row[6]))

#print(start_lng,start_lat,end_lng,end_lat,start_timestamp,duration)
"""
all_lng = start_lng[:]
all_lng.extend(end_lng)

all_lat = start_lat[:]
all_lat.extend(end_lat)

mean_lng = np.mean(all_lng)
mean_lat = np.mean(all_lat)
max_lng = max(all_lng)
max_lat = max(all_lat)
min_lng = min(all_lng)
min_lat = min(all_lat)

scale_lng = max_lng - min_lng + 0.001
scale_lat = max_lat - min_lat + 0.001

new_lng = [(c-min_lng)/scale_lng for c in all_lng]
new_lat = [(c-min_lat)/scale_lat for c in all_lat]

# print(new_lng,new_lat)
"""

mean_lng_start = np.mean(start_lng)
mean_lat_start = np.mean(start_lat)
max_lng_start = max(start_lng)
max_lat_start = max(start_lat)
min_lng_start = min(start_lng)
min_lat_start = min(start_lat)

mean_lng_end = np.mean(end_lng)
mean_lat_end = np.mean(end_lat)
max_lng_end = max(end_lng)
max_lat_end = max(end_lat)
min_lng_end = min(end_lng)
min_lat_end = min(end_lat)

scale_lng_start = max_lng_start - min_lng_start + 0.001
scale_lat_start = max_lat_start - min_lat_start + 0.001

scale_lng_end = max_lng_end - min_lng_end + 0.001
scale_lat_end = max_lat_end - min_lat_end + 0.001

new_lng_start = [(c-min_lng_start)/scale_lng_start for c in start_lng]
new_lat_start = [(c-min_lat_start)/scale_lat_start for c in start_lat]

new_lng_end = [(c-min_lng_end)/scale_lng_end for c in end_lng]
new_lat_end = [(c-min_lat_end)/scale_lat_end for c in end_lat]

plt.figure(1)
plt.plot(new_lng_start, new_lat_start, 'or', color = 'red')
plt.show()

plt.figure(2)
plt.plot(new_lng_end, new_lat_end, 'or', color = 'blue')
plt.show()


start_points = np.dstack((new_lng_start, new_lat_start))[0]
end_points = np.dstack((new_lng_end, new_lat_end))[0]

start_points = StandardScaler().fit_transform(start_points)
end_points = StandardScaler().fit_transform(end_points)
#print (start_points)
#print (end_points)

db = DBSCAN(eps=0.8, min_samples=8).fit(start_points)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels))
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(start_points, labels))
# Black removed and is used for noise instead.

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)
    #print (class_member_mask)
    #print (core_samples_mask)

    xy = start_points[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = start_points[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

db = DBSCAN(eps=0.8, min_samples=8).fit(end_points)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)
    #print (class_member_mask)
    #print (core_samples_mask)

    xy = end_points[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = end_points[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

