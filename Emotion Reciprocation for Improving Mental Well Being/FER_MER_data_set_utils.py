from oulu_casia_utils import (oulu_casia_get_data_set,
                              oulu_casia_expand_sequences,
                              _emotion_label_to_idx)
import numpy as np
from keras.utils import to_categorical
from copy import copy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


_oulu_casia_ds_mode = {
        'sequence',
        'expanded',
        'modified_expanded'
        }


_oulu_casia_config = {
                '_images_root_path' : '/home/varsrao/Downloads/OriginalImg',
                '_max_im_per_seq' : 9,
                '_image_resolution' : (240, 240),
                }


_oulu_casia_train_set_data_gen_config = {
                'rotation_range' : 15,
                'width_shift_range' : 0.1,
                'height_shift_range' : 0.1,
                'horizontal_flip' : True
                }


_oulu_casia_test_set_data_gen_config = {
                'rotation_range' : 20,
                'width_shift_range' : 0.2,
                'height_shift_range' : 0.2,
                'horizontal_flip' : True
                }


class oulu_casia_ds(object):
    
    def __init__(self,
                 dataset_mode = 'sequence',
                 data_set_config = _oulu_casia_config,
                 train_data_gen_config = _oulu_casia_train_set_data_gen_config,
                 test_data_gen_config = _oulu_casia_test_set_data_gen_config,
                 test_set_fraction = 0.1,
                 normalize_and_center = False):
        '''
        Input 1: OULU CASIA Data Set Mode (Default: sequence)
        Input 2: Data Set Configuration Dictionary (Default Included)
        Input 3: Image Data Generator Dictionary (Default Included)
        Input 4: Fraction of dataset dedicated for test set (Default: 0.1)
        Input 5: Normalize and Center Images Flag (Default: False)
        Purpose: Initialize OULU CASIA Data Set
        Output: None
        '''
        # Sanity Check
        if dataset_mode not in _oulu_casia_ds_mode:
            raise Exception('Invalid Data Set Mode')
        if test_set_fraction > 1 or test_set_fraction < 0:
            raise Exception('Invalid Test Set Fraction')
        
        # Configure instance
        self._oulu_casia_config = data_set_config
        self._oulu_casia_train_set_data_gen_config = train_data_gen_config
        self._oulu_casia_test_set_data_gen_config = test_data_gen_config
        self.dataset_mode = dataset_mode
        
        # Get raw dataset
        if dataset_mode == 'sequence':
            X, y = self.get_ds_as_sequence()
        elif dataset_mode == 'expanded':
            X, y = self.get_ds_as_expanded()
        else:
            X, y = self.get_ds_as_modified_expanded()
        
        # Perform training and test set split
        self.X_train, self.X_test, self.y_train, self.y_test = (
                train_test_split(X,
                                 y,
                                 test_size = test_set_fraction))
        # Normalize and center images if flag is set
        if normalize_and_center:
            self.normalize_and_center_images()

    
    def get_ds_as_sequence(self):
        '''
        Input: None
        Purpose: Wrapper for oulu_casia_get_data_set
        Output: Output of oulu_casia_get_data_set
        '''
        X, y = oulu_casia_get_data_set(**self._oulu_casia_config)
        '''
                _images_root_path = self._oulu_casia_config[
                        'images_root_path'],
                _max_im_per_seq = self._oulu_casia_config['max_im_per_seq'],
                _image_resolution = self._oulu_casia_config['image_resolution']
                )
        '''
        return [X, y]
    
    
    def get_ds_as_expanded(self):
        '''
        Input: None
        Purpose: Wrapper for oulu_casia_expand_sequences
        Output: Output of oulu_casia_get_data_set
        '''
        X, y = self.get_ds_as_sequence()
        X, y = oulu_casia_expand_sequences(X, y)
        return [X, y]
    
    
    def get_ds_as_modified_expanded(self):
        '''
        Input: None
        Purpose: Modify the expanded OULU CASIA dataset to extract only some of
                 the images of the sequence and custom label the extracted
                 images
        Output: [Extracted Images, Modified Labels]
        '''
        _max_im_per_seq = self._oulu_casia_config['_max_im_per_seq']
        mask = np.array([False for i in range(_max_im_per_seq)])
        # 1st image as "Neutral" emotion
        mask[0] = True
        # Last 5 images representing labelled emotion
        mask[-5:] = True
        # Modify 1st label to be "Neutral Expression"
        modified_labels = {
                # Index : Emotion
                0 : 'Neutral'
                }
        X, y = self.get_ds_as_expanded()
        new_images = []
        new_labels = []
        # Go through each sequence, mask the sequence and modify the labels
        for i in range(len(X) // _max_im_per_seq):
            start = i * _max_im_per_seq
            end = start + _max_im_per_seq
            sequence = X[start:end]
            sequence = sequence[mask]
            sequence_labels = y[start:end]
            sequence_labels = sequence_labels[mask]
            for index, modified_label in modified_labels.items():
                sequence_labels[index] = modified_label
            new_images.extend(sequence)
            new_labels.extend(sequence_labels)
        new_images = np.array(new_images)
        new_labels = np.array(new_labels)
        return [new_images, new_labels]
        
    
    def labels_to_categorical(self):
        '''
        Input: None
        Purpose: Convert the emotion labels to one hot encoded values
        Output: None
        '''
        emotion_labels_copy = copy(_emotion_label_to_idx)
        neutral_included = (True if self.dataset_mode == 'modified_expanded'
                            else False)
        if neutral_included:
            emotion_labels_copy['Neutral'] = len(_emotion_label_to_idx)
        num_classes = len(emotion_labels_copy)
        # Modify Training Labels
        self.y_train = [emotion_labels_copy[label] for label in 
                            self.y_train]
        self.y_train = to_categorical(self.y_train, num_classes)
        
        # Modify Testing Labels
        self.y_test = [emotion_labels_copy[label] for label in 
                            self.y_test]
        self.y_test = to_categorical(self.y_test, num_classes)
        
    
    def get_train_set_data_generator(self):
        '''
        Input: None
        Purpose: Returns a ImageDataGenerator fit to the training set
                 configured according to the OULU CASIA Data Gen configuration
        Output: ImageDataGenerator object
        '''
        datagen = ImageDataGenerator(
                **self._oulu_casia_train_set_data_gen_config)
        datagen.fit(self.X_train)
        return datagen
    
    
    def get_test_set_data_generator(self):
        '''
        Input: None
        Purpose: Returns a ImageDataGenerator fit to the testing set
                 configured according to the OULU CASIA Data Gen configuration
        Output: ImageDataGenerator object
        '''
        datagen = ImageDataGenerator(
                **self._oulu_casia_test_set_data_gen_config)
        datagen.fit(self.X_test)
        return datagen
    

    def normalize_and_center_images(self):
        '''
        Input: None
        Purpose: Subract images values by the mean and divide the difference by
                 the standard deviation
        Output: None
        '''
        self.X_train = ((self.X_train - self.X_train.mean(axis=0)) /
                        self.X_train.std(axis=0))
        self.X_test = ((self.X_test - self.X_test.mean(axis=0)) /
                       self.X_test.std(axis=0))

        
    
