import argparse

from exploremyai.utils.pattern_generator import create_floats_pattern_timeseries
from exploremyai.utils.data_sets import get_supervised_timeseries_data_set
from exploremyai.utils.models import create_lstm_regressor, train_model


def get_cli_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('predictor', choices=['regressor', 'classifier'])
    parser.add_argument('--pattern_size',
                        required=True,
                        type=int,
                        help='Sequence length')
    parser.add_argument('--batch_size',
                        required=True,
                        type=int)
    parser.add_argument('--max_epochs',
                        default=100000,
                        type=int)
    parser.add_argument('--lstm_cells',
                        default=4,
                        type=int)
    parser.add_argument('--k',
                        default=50,
                        type=int,
                        help='Number of times the sequence is repeated in the timeseries training data')
    parser.add_argument('--min_value',
                        default=-1000,
                        type=int,
                        help='Minimum value in the sequence')
    parser.add_argument('--max_value',
                        default=-1000,
                        type=int,
                        help='Maximum value in the sequence')
    parser.add_argument('--input_timesteps',
                        type=int,
                        help='Explicit input window size for LSTM, default is pattern_size - 1')
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    return parser.parse_args()


def run_lstm_exploration(pattern_size, batch_size, max_epochs, lstm_cells, k, min_value, max_value,
        input_timesteps=None, input_file=None, output_file=None):
    # # ----------------------------------------------------------------------------------------------------------- # #
    # # -------------------------------------------------- SETUP -------------------------------------------------- # #
    # # ----------------------------------------------------------------------------------------------------------- # #
    sequence_size = pattern_size
    if input_timesteps:
        window_size = input_timesteps
    else:
        window_size = sequence_size - 1
    # # ----------------------------------------------------------------------------------------------------------- # #
    # # --------------------------------------------- GET TIMESERIES ---------------------------------------------- # #
    # # ----------------------------------------------------------------------------------------------------------- # #
    series_length = (k + 1) * sequence_size - 1
    timeseries = create_floats_pattern_timeseries(series_length, sequence_size, min_value, max_value,
        from_file=input_file, to_file=output_file)
    # # ----------------------------------------------------------------------------------------------------------- # #
    # # --------------------------------------------- CREATE DATASET ---------------------------------------------- # #
    # # ----------------------------------------------------------------------------------------------------------- # #
    X, y = get_supervised_timeseries_data_set(timeseries, window_size)
    X, y = X.values.reshape(-1, window_size, 1), y.values
    # # ----------------------------------------------------------------------------------------------------------- # #
    # # -------------------------------------------- CREATE LSTM MODEL -------------------------------------------- # #
    # # ----------------------------------------------------------------------------------------------------------- # #
    model = create_lstm_regressor(lstm_cells, window_size)
    print(model.summary())
    # # ----------------------------------------------------------------------------------------------------------- # #
    # # ----------------------------------------------- TRAIN MODEL ----------------------------------------------- # #
    # # ----------------------------------------------------------------------------------------------------------- # #
    model, _ = train_model(model, X, y, batch_size=batch_size, epochs=max_epochs, verbose=0)
    model.save('keras_'+str(sequence_size)+'.model')
    # # ----------------------------------------------------------------------------------------------------------- # #
    # # ------------------------------------------------ EVALUATE ------------------------------------------------- # #
    # # ----------------------------------------------------------------------------------------------------------- # #
    loss = model.evaluate(X, y, batch_size=batch_size) 
    print('[evaluation] MSE:', loss)
    return loss


def main():
    args = get_cli_config()
    run_lstm_exploration(
        pattern_size=args.pattern_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lstm_cells=args.lstm_cells,
        k=args.k,
        min_value=args.min_value,
        max_value=args.max_value,
        input_timesteps=args.input_timesteps,
        input_file=args.input_file,
        output_file=args.output_file,
        )


if __name__ == "__main__":
    main()