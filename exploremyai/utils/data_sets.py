import pandas as pd
import numpy as np


def get_supervised_timeseries_data_set(data, input_steps):
    """This function transforms a univariate timeseries into a supervised learning problem where the input consists
    of sequences of length input_steps and the output is the prediction of the next step
    """
    series = pd.Series(data)
    data_set = pd.DataFrame({'t' : series, 't+1' : series.shift(-1)})
    if input_steps > 1:
        x_values = np.concatenate([data[i:i+input_steps]
                                    .reshape(1, input_steps) for i in range(len(series) - input_steps)])
        timesteps_df = pd.DataFrame(x_values[:,:-1], index=np.arange(input_steps - 1, input_steps - 1 + len(x_values)),
                                    columns = ['t-' + str(input_steps - i) for i in range(1, input_steps)])
        data_set = pd.concat([timesteps_df, data_set], axis=1, join='inner')
    data_set = data_set.dropna()
    X = data_set.drop('t+1', axis=1)
    y = data_set.loc[:,'t+1']
    return (X, y)