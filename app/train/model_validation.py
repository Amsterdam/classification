import csv
import re
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV


def calculate_metrics(test_labels: pd.Series, test_predict: Iterable) \
        -> tuple[float, float, float]:
    """
    Calculate precision, recall and accuracy metrics of a binary classification.

    Parameters
    ----------
    test_labels : pd.Series
        The ground truth labels.
    test_predict : Iterable
        The predicted labels returned by GridSearchCV.predict().

    Returns
    -------
    tuple[float, float, float]
        A tuple containing precision, recall and accuracy, respectively.
    """
    precision = precision_score(test_labels,
                                test_predict,
                                average='macro',
                                zero_division=0)
    recall = recall_score(test_labels,
                          test_predict,
                          average='macro')
    accuracy = accuracy_score(test_labels,
                              test_predict)
    return precision, recall, accuracy


def generate_and_save_confusion_matrix(model: GridSearchCV,
                                       test_texts: pd.Series,
                                       test_labels: pd.Series,
                                       pdf_filepath: str,
                                       csv_filepath: str):
    """
    Generate a confusion matrix from a trained model and save as image and CSV.

    Parameters
    ----------
    model : GridSearchCV
        The trained model which predict method will be used.

    test_texts : pd.Series
        A pandas series of the test data to use for generating the confusion matrix.

    test_labels : pd.Series
        The actual labels for the test data.

    pdf_filepath : str
        The file path where the confusion matrix should be saved.

    csv_filepath : str
        The file path where the confusion matrix as CSV should be saved.

    Returns
    -------
    None
    """
    plt.rcParams["figure.figsize"] = (30, 30)
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        test_texts,
        test_labels,
        cmap=plt.cm.Blues,
        normalize=None,
        xticks_rotation="vertical"
    )
    plt.savefig(pdf_filepath)
    df2 = pd.DataFrame(disp.confusion_matrix, columns=disp.display_labels)
    df2.to_csv(csv_filepath)


def save_detailed_validation_results(test_texts: pd.Series,
                                     test_predict: pd.Series,
                                     test_labels: pd.Series,
                                     csv_filepath: str):
    """
    This function iterates through all the predictions and labels, finds the
    mismatched ones and writes them into a CSV file. In the CSV file, each row
    will contain the text, predicted category, and actual category.

    Parameters
    ----------
    test_texts : pd.Series
        A pandas series containing the test data texts.

    test_predict : pd.Series
        A pandas series containing the predicted labels.

    test_labels : pd.Series
        A pandas series containing the actual labels for the test data.

    csv_filepath : str
        The file path where the CSV file should be saved.

    Returns
    -------
    None
    """
    with open(csv_filepath, 'w') as csvfile:
        fieldnames = ['text', 'predicted_category', 'actual_category']
        writer = csv.DictWriter(csvfile,
                                fieldnames=fieldnames,
                                quoting=csv.QUOTE_NONNUMERIC)

        writer.writeheader()
        for _input, prediction, label in zip(test_texts, test_predict, test_labels):
            if prediction != label:
                writer.writerow(
                    {
                        'text': re.sub(pattern=r"\W", repl=" ", string=_input),
                        'predicted_category': prediction,
                        'actual_category': label
                    }
                )
