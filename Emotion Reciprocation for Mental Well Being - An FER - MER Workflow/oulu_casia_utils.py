import os
import numpy as np
from PIL import Image
import re
from math import log2

_image_conditions = {
        'NI',
        'VL'
        }

_lighting_conditions = {
        'Dark',
        'Strong',
        'Weak'
        }

_emotion_labels = {
        'Anger',
        'Disgust',
        'Fear',
        'Happiness',
        'Sadness',
        'Surprise'
        }

_emotion_label_to_idx = {
        label:index for index, label in enumerate(_emotion_labels)}

_image_extension = 'jpeg'
_image_width = 320
_image_height = 240


def _reduce_sequence_len(_sequence_list,
                        _max_elem_per_seq,
                        delete_even_first):
    '''
    Input 1: Sequence List
    Input 2: Maximum Elements Per Sequence
    Input 3: Delete Even First Flag
    Purpose: Repetitively delete alternatively odd/even indices until length
             of sequence is Max Elements Per Sequence
    Output: List Containing Max Elements Per Sequence
    '''
    _len_seq = len(_sequence_list)
    if _len_seq <= _max_elem_per_seq:
        return _sequence_list
    # Since we reduce the size of sequence by 2 by deleting alternate numbers,
    # Max Elements Per Sequence = Length of Sequence / (2 ^ Min Num Passes)
    _min_num_passes = int(log2(_len_seq // _max_elem_per_seq))
    new_sequence = _sequence_list
    for i in range(_min_num_passes):
        new_sequence = (
                new_sequence[0::2] if delete_even_first
                else new_sequence[1::2])
        delete_even_first = not delete_even_first
    # Delete remainining elements in alternate fashion as well
    rem_to_del = len(new_sequence) - _max_elem_per_seq
    mask = np.array([True for i in range(len(new_sequence))])
    del_from = 0 if delete_even_first else 1
    for i in range(rem_to_del):
        mask[del_from] = False
        del_from += 2
    new_sequence = new_sequence[mask]
    return new_sequence
        
    
def oulu_casia_get_data_set(_images_root_path,
                            _image_condition = 'VL',
                            _lighting_condition = 'Strong',
                            _max_im_per_seq = float('inf'),
                            _image_resolution = (_image_width, _image_height),
                            _return_min_sequence = False
                            ):
    '''
    Input 1: Path of OriginalImg/Preprocess Img (Set B or C)
    Input 2: Image Condition Required (Default: VL)
    Input 3: Lighting Condition Required (Default: Strong)
    Input 4: Max Images Per Sequence (Default: All)
    Input 5: Image Resolution in (Width, Height) (Default: Original)
    Input 6: Flag to return Minimum Length of All Sequences (Default: False)
    Purpose: Get OULU CASIA Dataset from original directory structure into 2
             lists denoting image sequences and the corresponding emotion label
    Output: [[[Image Sequences]], [Emotion Labels], <Min Length>]
    '''
    # Validate Arguments
    contents = set(os.listdir(_images_root_path))
    if set.intersection(contents, _image_conditions) != _image_conditions:
        raise Exception('Invalid Path Passed')
    if _image_condition not in _image_conditions:
        raise Exception('Invalid Image Condition Passed')
    if _lighting_condition not in _lighting_condition:
        raise Exception('Invalid Lighting Condition Passed')
        
    # Crawl and retrieve data
    next_path = _images_root_path + '/' + _image_condition
    next_path += '/' + _lighting_condition
    sequence_id = 0
    sequences = {}
    min_sequence = float('inf')
    _file_extension = re.compile('.*\.(.*)')
    for person in os.listdir(next_path):
        person_path = next_path + '/' + person
        emotion_dirs = set(os.listdir(person_path))
        # Handle missing emotion images
        if set.intersection(emotion_dirs, _emotion_labels) != _emotion_labels:
            missing_emotion = _emotion_labels - emotion_dirs
            raise Exception('Emotions {0} missing from {1}'.format(
                    missing_emotion, person_path))
        # Construct image sequence per emotion
        for emotion in _emotion_labels:
            emotion_path = person_path + '/' + emotion
            images_list = sorted(os.listdir(emotion_path))
            image_sequence = []
            for image_name in images_list:
                image_path = emotion_path + '/' + image_name
                extension = _file_extension.findall(image_name)[0]
                if extension == _image_extension:
                    image = Image.open(image_path)
                    if _image_resolution != (_image_width, _image_height):
                        image = image.resize(_image_resolution)
                    image_arr = np.array(image)
                    image_sequence.append(image_arr)
            image_sequence = np.array(image_sequence)
            # Useful: Keep track of length of smallest sequence
            if len(image_sequence) < min_sequence:
                min_sequence = len(image_sequence)
            # Summarize sequence by smartly reducing effective sequence length
            image_sequence = _reduce_sequence_len(image_sequence,
                                                 _max_im_per_seq, True)
            # Pack image sequence and corresponding emotion
            sequences[sequence_id] = [image_sequence, emotion]
            sequence_id += 1
    # Unpack and form association
    image_sequences = np.array(
            [seq_data[0] for seq_id, seq_data in sequences.items()])
    emotions = np.array(
            [seq_data[1] for seq_id, seq_data in sequences.items()])
    return_list = [image_sequences, emotions]
    if _return_min_sequence:
        return_list.append(min_sequence)
    return return_list


def oulu_casia_expand_sequences(image_sequences, seq_labels):
    '''
    Input 1: Image Sequences as a 5D numpy array
    Input 2: Sequence Labels as a numpy array
    Purpose: Expand sequence into elements and assign each element the
             sequence label
    Output: [Images, Labels]
    '''
    _im_seq_shape = image_sequences.shape
    if len(_im_seq_shape) != 5:
        raise Exception('Expected 5D Input, received {0}D', len(_im_seq_shape))
    if len(seq_labels) != _im_seq_shape[0]:
        raise Exception(
                'Number of Sequences({0}) not matching Number of Labels({1})'.
                format(_im_seq_shape[0], len(seq_labels)))
    image_list = image_sequences.reshape(
            (_im_seq_shape[0] * _im_seq_shape[1], *_im_seq_shape[2:]))
    image_labels = np.repeat(seq_labels, _im_seq_shape[1])
    return [image_list, image_labels]


if __name__ == '__main__':
    images_root_path = input(
            '*** Enter path for OULU CASIA Original/Preprocess Img ***\n')
    X, Y = oulu_casia_get_data_set(
            _images_root_path = images_root_path,
            _max_im_per_seq = 9)
    print('Data Shape: {0}, Labels Shape: {1}'.format(
            X.shape, Y.shape))
    X, Y = oulu_casia_expand_sequences(X, Y)
    print('After Expansion:')
    print('Data Shape: {0}, Labels Shape: {1}'.format(
            X.shape, Y.shape))