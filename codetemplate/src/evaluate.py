"""**Evaluate New Data**

----
"""
import sys
import getopt
import pickle
import pandas as pd

from data_processing import preprocess_dataset, generate_evaluation_data


def evaluate_isolation_forest(X, modelfile, filepath=''):
    """ Evaluate a trained isolation forest to the given data.

    :param X: Evaluation data with m features and n training points.
    :type X: numpy array (m x n)
    :param modelfile: Filepath to the trained data.
    :type modelfile: string
    :param filepath: (optional) Filepath to save the prediction.
    :type filepath: string

    :param modelfile: Filepath to the trained data.
    :type modelfile: string


    :return result: Prediction values (-1 outlier, 1 normal).
    :rtype: numpy array"""
    model = pickle.load(open(modelfile, 'rb'))
    result = model.predict(X)

    if filepath != '':
        pd.DataFrame(result, columns=["Prediction"]).to_csv(filepath, index=False)

    return result


def main(argv):
    """
    The ML evaluation steps including:
        * reshuffling of the sensor data (combination of identical time stamps)
        * loading of trained model
        * evaluation of the data with loaded isolation forest
    """

    inputfile = ''
    outputfile = ''
    try:
        opts, _ = getopt.getopt(argv, "h:i:m:o:", ["ifile=", "model=", "ofile="])
    except getopt.GetoptError:
        print('data_processing.py -i <inputfile> -m <modelfile> -o <modelfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('data_processing.py -i <inputfile> -m <modelfile> -o <modelfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-m", "--model"):
            modelfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    data = preprocess_dataset(inputfile)
    X = generate_evaluation_data(data)

    _ = evaluate_isolation_forest(X, modelfile, outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
