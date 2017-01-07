#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from numpy import *

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
        if cnt > 100:
            break
        cnt += 1
        print (row)
        start_lng.append(float(row[1]))
        start_lat.append(float(row[2]))
        end_lng.append(float(row[3]))
        end_lat.append(float(row[4]))
        start_timestamp.append(int(row[5]))
        duration.append(int(row[6]))

mean_lng_start = mean(start_lng)
mean_lat_start = mean(start_lat)
max_lng_start = max(start_lng)
max_lat_start = max(start_lat)
min_lng_start = min(start_lng)
min_lat_start = min(start_lat)

mean_lng_end = mean(end_lng)
mean_lat_end = mean(end_lat)
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

print(new_lng_end, new_lat_end)
