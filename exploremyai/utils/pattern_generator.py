import os

import numpy as np
import pandas as pd


def generate_floats_sequence(sequence_size, min_value, max_value):
    """This function generates a list of random floating point numbers
    of length sequence_size and range [min_value, max_value)
    """
    seq = min_value + ((max_value - min_value) * np.random.rand(sequence_size))
    return seq


def generate_integers_sequence(sequence_size, min_value, max_value):
    """This function generates a list of random integer numbers
    of length sequence_size and range [min_value, max_value)
    """
    seq = np.random.randint(min_value, max_value, sequence_size)
    return seq


def create_floats_pattern_timeseries(series_length, sequence_size, min_value, max_value,
        from_file=None, to_file=None):
    """This function returns a timeseries of length series_length formed by a repetition of patterns with length
    sequence_size. It will also save the pattern to a file if a file location is provided at to_file
    """
    if from_file and os.path.exists(from_file):
        regression_pattern = pd.read_csv(from_file).values
        if len(regression_pattern.shape) > 1:
            assert regression_pattern.shape[1] == 1, \
                'A file with multiple columns was provided, multivariate series are not supported'
        if len(regression_pattern) > sequence_size:
            regression_pattern = np.concatenate(regression_pattern,
                generate_floats_sequence(sequence_size - len(regression_pattern), min_value, max_value))
    else:
        regression_pattern = generate_floats_sequence(sequence_size, min_value, max_value)
    if to_file:
        pd.DataFrame({'Random_pattern' : regression_pattern}).to_csv(to_file, index=False)
    series = np.resize(regression_pattern[:sequence_size], series_length)
    return series
