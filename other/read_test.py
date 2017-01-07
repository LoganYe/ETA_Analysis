#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
with open('../data/test.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    cnt = 0
    for row in spamreader:
        if 0 == cnt:
            cnt += 1
            continue
        if cnt > 10000:
            break
        cnt += 1
        print (row)