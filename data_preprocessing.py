import numpy as np
import pandas as pd
import os
from glob import iglob
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_iterators(batch_size, buffer_size=10000):
    """
    
    """
    rootdir = '/home/mlisfun/SharpestMinds/FeaturedProject-EEG/data_tfrecords/'
    d = get_TFRecords_files_dict()
    filenames = []
    for key in d:
        for file_name in d[key]:
            filenames.append(rootdir + key + '/' + file_name)
    filenames.sort()
    train_filenames, other_filenames = train_test_split(filenames, test_size=0.2, random_state=42)
    val_filenames, test_filenames = train_test_split(other_filenames, test_size=0.5, random_state=42)
    
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    val_dataset = tf.data.TFRecordDataset(val_filenames)
    test_dataset = tf.data.TFRecordDataset(test_filenames)
   
    train_dataset, train_iterator = _dataset_process(train_dataset, batch_size, buffer_size)
    val_dataset, val_iterator = _dataset_process(val_dataset, batch_size, buffer_size)
    test_dataset, test_iterator = _dataset_process(test_dataset, batch_size, buffer_size)
    
    return train_iterator, val_iterator, test_iterator

    
def _dataset_process(dataset, batch_size, buffer_size=10000):
    """
    
    """
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    
    return dataset, iterator


    
def _parse_function(example_proto):
    """
    
    """
   
    keys = ['P2', 'X', 'F2', 'FZ', 'FC6', 'P7', 'FPZ', 'FT8', 'P5', 'Y', 'F3', 
            'CPZ', 'TP7', 'C6', 'CP4', 'FP1', 'F6', 'C3', 'CP2', 'FC3', 'C2', 
            'P3', 'F4', 'C1', 'nd', 'CP5', 'FP2', 'F8', 'O2', 'CP1', 'P4', 'PO2', 
            'OZ', 'CZ', 'AFZ', 'AF2', 'PO1', 'TP8', 'POZ', 'FCZ', 'T8', 'FC1', 
            'F5', 'CP3', 'CP6', 'FT7', 'F7', 'P1', 'T7', 'PZ', 'F1', 'FC4', 
            'PO7', 'O1', 'PO8', 'P8', 'FC2', 'FC5', 'C5', 'P6', 'AF1', 'AF8', 'C4', 'AF7']
    keys.sort()
    
    features = {"label": tf.FixedLenFeature((), tf.int64, default_value=0)}

    for key in keys:
        features[key] = tf.FixedLenFeature((), tf.string, default_value="")
    
    
    parsed = tf.parse_single_example(example_proto, features)
    
    feature_array = []
    for key in keys:
        feature = tf.decode_raw(parsed[key], tf.float32)
        feature_array.append(feature)

    feature_array = tf.expand_dims(feature_array, -1)
    
    label = tf.cast(parsed["label"], tf.int32)
    label = tf.one_hot(label, 2)
    
    return {'label': label, 'feature': feature_array}


def get_file_as_dict(folder_name='co2c0000378', file_name='co2c0000378.rd.037'):
    """
    
    """
    rootdir = '/home/mlisfun/SharpestMinds/FeaturedProject-EEG/eeg_full'
    file_loc = rootdir + '/' + folder_name + '/' + file_name

    df = pd.read_csv(file_loc, sep=' ', skiprows=4)
    df.drop(['#'], axis=1)
    df = df.drop(df.index[df['#'] == '#'])
    
    dict_ = {}
    for ent in set(df['FP1']):
        dict_[ent] = df[df['FP1'] == ent]['0'].values
    
    label = folder_name[3]
    subject_num = folder_name[-3:]
    trial_num = file_name[-3:]
    
    matchcond_df = pd.read_csv(file_loc, sep=' ', skiprows=3, nrows = 1, header=None)
    matchcond = matchcond_df[1].values[0] + matchcond_df[2].values[0]
    
    dict_['label'] = label
    dict_['subject_num'] = subject_num
    dict_['trial_num'] = trial_num
    dict_['matchcond'] = matchcond
    
    return dict_

def get_files_dict():
    """
    
    """
    rootdir = '/home/mlisfun/SharpestMinds/FeaturedProject-EEG/eeg_full'
    d = {}
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            sub_num = subdir[-11:]
            file_name = file
            if sub_num in d:
                d[sub_num].append(file_name)
            else:
                d[sub_num] = [file_name]

    return d

def get_TFRecords_files_dict():
    """
    
    """
    rootdir = '/home/mlisfun/SharpestMinds/FeaturedProject-EEG/data_tfrecords'
    d = {}
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            sub_num = subdir[-11:]
            file_name = file
            if sub_num in d:
                d[sub_num].append(file_name)
            else:
                d[sub_num] = [file_name]

    return d

def generate_TFRecords():
    """Generates all of the TFRecords files."""
    print('Generating...')
    files_dict = get_files_dict()
    for sub_num in files_dict:
        for file_name in files_dict[sub_num]:
            try:
                d = get_file_as_dict(sub_num, file_name)
                save_to_TFRecords(d, sub_num, file_name)
            except:
                print('Error loading ', file_name)
                continue
    print('Generation Complete.')
    
def save_to_TFRecords(d, sub_name, file_name, overwrite=False):
    """Saves feature and label data to TFRecords format and 
    writes it to data_tfreconds/filname.tfrecords.
    
    :input d: the dictionary
    
    :sub_name: the name of the subfolder to write to
    :input file_name: the filename of the file to write
    """
    filename = os.path.join('data_tfrecords/' + sub_name + '/' + file_name + '.tfrecords')
    if (not os.path.exists(filename)) and (overwrite == False):
        print('Preparing', filename)

        label = d['label']
        del d['label']
        if label == 'a':
            label = 1
        elif label == 'c':
            label = 0

        subject_num = d['subject_num']
        del d['subject_num']
        trial_num = d['trial_num']
        del d['trial_num']

        matchcond = d['matchcond']
        del d['matchcond']
        if matchcond[:4] == 'S1ob':
            matchcond = 0
        elif matchcond[:4] == 'S2ma':
            matchcond = 1
        elif matchcond[:4] == 'S2no':
            matchcond = 2
        else:
            raise ValueError('matchcond not one of S1obj, S2match, or S2nomatch, ' +
                             'had value |{}|.'.format(matchcond))
        # Create directory if needed
        if not os.path.exists('data_tfrecords/' + sub_name):
            os.makedirs('data_tfrecords/' + sub_name)


        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        feature={
            'subject_num': _int64_feature(int(subject_num)),
            'trial_num': _int64_feature(int(trial_num)),
            'matchcond': _int64_feature(matchcond),
            'label': _int64_feature(label)}
        for key in d:
            feature_float = np.float32(d[key])
            feature_raw = feature_float.tostring()
            feature[key] = _bytes_feature(feature_raw)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        writer.close()
#     else:
#         print('File exists.')
        
def _int64_feature(value):
    """Helper function to create int64 tensorflow feature object.
    
    :input value: the value of the feature.
    :returns: the feature object
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Helper function to create bytes list tensorflow feature object.
    
    :input value: the value of the feature.
    :returns: the feature object"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))