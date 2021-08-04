"""**Raw Data Handling**

----
"""
import sys
import getopt
import csv
import pandas as pd
import numpy as np


def preprocess_dataset(input_data, save_data=False):
    """ The preprocessing selects the relevant data.

    :param input_data: File path for input data (csv-file from temperature sensor)
    :type input_data: string

    :return df: Transformed data containing the training data
                (datetime, heating temperature, cooling temperature).
    :rtype: pandas dataframe"""

    source_id = ""
    start_date = ""
    end_date = ""
    with open(input_data, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        new_table = []
        temp_line = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                if source_id == "":
                    source_id = row[0]
                if start_date == "":
                    start_date = row[1][:10]
                end_date = row[1][:10]

                # merge data with same timestamp
                if temp_line == []:
                    if row[2] == "heating_temperature":
                        temp_line = [row[1], row[3], np.nan]
                    else:
                        temp_line = [row[1], np.nan, row[3]]
                else:
                    if temp_line[0] == row[1]:
                        if row[2] == "heating_temperature":
                            temp_line[1] = row[3]
                        else:
                            temp_line[2] = row[3]
                        new_table.append(temp_line)
                        temp_line = []
                    else:
                        new_table.append(temp_line)
                        if row[2] == "heating_temperature":
                            temp_line = [row[1], row[3], np.nan]
                        else:
                            temp_line = [row[1], np.nan, row[3]]
                line_count += 1
        print(f'Processed {line_count} lines.')

    df = pd.DataFrame(new_table, columns=['datetime', 'heating_temperature', 'cooling_temperature'])
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%dT%H:%M:%S.000+0000")
    df["heating_temperature"] = pd.to_numeric(df["heating_temperature"])
    df["cooling_temperature"] = pd.to_numeric(df["cooling_temperature"])

    if save_data:
        df.to_csv("../../data/" + source_id + "_" + start_date + "_" + end_date + ".csv",
                  index=False)

    return df


def generate_evaluation_data(input_data, na_heating=51, na_cooling=-11, max_t=20000):
    """ Converts the data for training/evaluation with IsolationForerst or DecisionTree.
        The datapoints at time t is combined with the previous datapoint.

    :param input_data: data from preprocessing (timestamp, heating temperature, cooling temperature)
    :type input_data: pandas DataFrame (3 x n dimension)

    :return X: Transformed data
               (delta time, heating temp., cooling temp., delta t-1, heat. t-1, cool. t-1).
    :rtype: numpy array (6 x n dimension)"""

    df = input_data.copy()

    # Replace NaN-values with constant values far from the relevant temperature
    # (but not too far ~3-5 std.-dev. distance)
    df['heating_temperature'] = df['heating_temperature'].fillna(na_heating)
    df['cooling_temperature'] = df['cooling_temperature'].fillna(na_cooling)

    # Convert time to time-difference
    df["delta_T"] = df["datetime"].diff() / np.timedelta64(1, 's')
    # Use information of previous data point
    df["hT_prev"] = df["heating_temperature"].shift(1)
    df["cT_prev"] = df["cooling_temperature"].shift(1)
    df["delta_T_prev"] = df['delta_T'].shift(1)

    # Fill NaN-values at the beginning with reasonable numbers
    df["hT_prev"] = df["hT_prev"].fillna(na_heating)
    df["cT_prev"] = df["cT_prev"].fillna(na_cooling)
    df['delta_T'] = df['delta_T'].fillna(max_t)
    df["delta_T_prev"] = df['delta_T_prev'].fillna(max_t)

    # Set a maximum time-interval to prevent interference of irrelevant outliers
    df['delta_T'].clip(upper=max_t)
    df['delta_T_prev'].clip(upper=max_t)

    return df[['delta_T', 'heating_temperature', 'cooling_temperature', 'delta_T_prev', 'hT_prev', 'cT_prev']].to_numpy()


def main(argv):
    """
    The ML data processling steps including:
        * reshuffling of the sensor data
    """
    inputfile = ''
    try:
        opts, _ = getopt.getopt(argv, "h:i:", ["ifile="])
    except getopt.GetoptError:
        print('data_processing.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('data_processing.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    preprocess_dataset(inputfile, save_data=True)


if __name__ == "__main__":
    main(sys.argv[1:])
