#!/usr/bin/env python2
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
"""
Functions for creating temporary LMDBs
"""

import argparse
import caffe
import csv
from collections import defaultdict
import h5py
import lmdb
import numpy as np
import os
import PIL.Image
import random
import re
import shutil
import sys
import time

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

DB_BATCH_SIZE = 1024

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
FEATURE_LEN = 1014

def create_lmdbs(folder, input_file_name, db_batch_size=None):
    """
    Creates LMDBs
    """

    if db_batch_size is None:
        db_batch_size = DB_BATCH_SIZE

    input_db = None
    # open output LMDB
    output_db = lmdb.open(folder, map_async=True, max_dbs=0)
    
    if input_file_name.endswith('.hdf5'):
        # open input HDF5
        input_db = h5py.File(input_file_name)    
        class_samples = []
        classes = input_db['classes'].keys()
        samples_per_class = None
        for c in classes:
            t = input_db['classes'][c]['data'][...]
            if samples_per_class is None:
                samples_per_class = t.shape[0]
            else:
                assert(samples_per_class == t.shape[0])
            class_samples.append(t)
    elif input_file_name.endswith('.csv'):
        # create character dict
        cdict = {}
        for i,c in enumerate(ALPHABET):
            cdict[c] = i
        samples = {}
        with open(input_file_name) as f:
            reader = csv.DictReader(f,fieldnames=['class'],restkey='fields')
            for row in reader:
                label = row['class']
                if label not in samples:
                    samples[label] = []
                sample = np.zeros(FEATURE_LEN)
                count = 0
                for field in row['fields']:
                    for char in field.lower():
                        if char in cdict:
                            sample[count] = cdict[char]
                        count += 1
                        if count >= FEATURE_LEN-1:
                            break
                samples[label].append(sample)
            samples_per_class = None
            classes = samples.keys()
            class_samples = []
            for c in classes:
                if samples_per_class is None:
                    samples_per_class = len(samples[c])
                else:
                    assert(samples_per_class == len(samples[c]))
                class_samples.append(samples[c])
    else:
        assert False, 'unknown extension'

    indices = np.arange(samples_per_class)
    np.random.shuffle(indices)

    print classes

    batch = []
    for idx in indices:
        for c,cname in enumerate(classes):
            sample = class_samples[c][idx].astype('uint8')
            sample = sample[..., np.newaxis, np.newaxis]
            datum = caffe.io.array_to_datum(sample, c)
            batch.append(('%d_%d' % (idx,c), datum))
        if len(batch) >= DB_BATCH_SIZE:
            _write_batch_to_lmdb(output_db, batch)
            batch = []

    # close databases
    if input_db is not None:
        input_db.close()
    output_db.close()

    return

def _write_batch_to_lmdb(db, batch):
    """
    Write a batch of (key,value) to db
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for key, datum in batch:
                lmdb_txn.put(key, datum.SerializeToString())
    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit*2
        print('Doubling LMDB map size to %sMB ...' % (new_limit>>20,))
        try:
            db.set_mapsize(new_limit) # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0,87):
                raise Error('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
            else:
                raise e
        # try again
        _write_batch_to_lmdb(db, batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create-LMDB tool')

    ### Positional arguments

    parser.add_argument('input',
            help='Input HDF5'
            )

    parser.add_argument('output',
            help='Output LMDB'
            )

    args = vars(parser.parse_args())

    if os.path.exists(args['output']):
        shutil.rmtree(args['output'])

    os.makedirs(args['output'])

    start_time = time.time()

    create_lmdbs(args['output'],
		         args['input'],
            )

    print 'Done after %s seconds' % (time.time() - start_time,)

