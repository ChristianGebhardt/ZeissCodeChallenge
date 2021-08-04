"""**The Machine Learning Development**

----
"""
import sys
import getopt
import pickle
import pandas as pd
from sklearn.ensemble import IsolationForest

from data_processing import preprocess_dataset, generate_evaluation_data


def train_isolation_forest(X, filename=''):
    """ Train an isolation forest to the given data.

    :param X: Training data with m features and n training points.
    :type X: numpy array (m x n)
    :param filena: Training data with m features and n training points.
    :type X: numpy array (m x n)

    :return df: Transformed data containing the training data
                (datetime, heating temperature, cooling temperature).
    :rtype: pandas dataframe"""

    model = IsolationForest(random_state=0).fit(X)

    result = model.predict(X)

    print(pd.DataFrame(result, columns=["Prediction"])["Prediction"].value_counts())

    if filename != '':
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    return model


def main(argv):
    """
    The ML training steps, including:
        * reshuffling of the sensor data (combination of identical time stamps)
        * training data generation (combine data from time t and t-1)
        * train and evaluate isolation forest
    """

    inputfile = ''
    outputfile = ''
    try:
        opts, _ = getopt.getopt(argv, "h:i:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('data_processing.py -i <inputfile> -o <modelfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('data_processing.py -i <inputfile> -o <modelfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    data = preprocess_dataset(inputfile)
    X = generate_evaluation_data(data)

    train_isolation_forest(X, outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
