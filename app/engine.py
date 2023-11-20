import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psutil
from nltk.stem.snowball import DutchStemmer
from settings import LOG_LEVEL
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

import nltk

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(level=LOG_LEVEL)
logger.addHandler(handler)


class TextClassifier:
    _text = 'Text'
    _main = 'Main'
    _middle = 'Middle'
    _sub = 'Sub'
    _lbl = 'Label'

    def __init__(self):
        """
        Initialize the TextClassifier instance.
        """
        self.model = None

        # Download stop words
        nltk.download('stopwords', raise_on_error=True)

        # Initialize the Dutch stemmer with optional stop word removal
        self.stemmer = DutchStemmer(ignore_stopwords=True)
        self.stop_words = nltk.corpus.stopwords.words('dutch')

    def pickle(self, obj: Any, file: str | Path):
        """
        Serialize and save an object to a file using joblib.

        Parameters
        ----------
        obj : obj
            The object to be serialized and saved.
        file : str, Path
            The file path where the object will be saved.
        """
        logger.info(f'Pickling "{type(obj).__name__}" to "{file}"')
        joblib.dump(obj, file)

    def export_model(self, file: str | Path):
        """
        Export the trained model to a file using joblib.

        Parameters
        ----------
        file : str, Path
            The file path where the model will be saved.
        """
        logger.info(f'Serializing model to "{file}"')
        joblib.dump(self.model, file)

    def preprocessor(self, text: str) -> str:
        """
        Preprocess the input text by converting it to lowercase, removing special characters,
        and applying stemming to each word.

        Parameters
        ----------
        text : str
            The input text to be preprocessed.

        Returns
        -------
        str
            The preprocessed text.
        """
        # Convert the text to lowercase
        text = text.lower()

        # Remove special characters using regular expression
        text = re.sub("\\W", " ", text)

        # Split the text into words
        words = re.split("\\s+", text)

        # Apply stemming to each word using the provided stemmer
        # and join the processed words back into a single string
        return ' '.join([
            self.stemmer.stem(word=word)
            for word in words
        ])

    def load_data(self, csv_file: str | Path, frac: float = 1) -> pd.DataFrame:
        """
        Load and preprocess data from a CSV file.

        Parameters
        ----------
        csv_file : str, Path
            The path to the CSV file containing the data.
        frac : float
            Fraction of the dataset to load. Default is 1, meaning the entire dataset.

        Returns
        -------
        pd.DataFrame
            A preprocessed DataFrame containing the loaded data.
        """
        # Read data from CSV file into a DataFrame
        df = pd.read_csv(csv_file, sep=None, engine='python')

        # Rename columns for clarity
        df.columns = [self._main, self._sub, self._text]

        # Drop rows with missing values in specified columns
        df = df.dropna(
            axis=0,
            how='any',
            subset=[self._main, self._sub, self._text],
            inplace=False,
        )

        # Remove duplicate rows based on the text column
        df = df.drop_duplicates(subset=[self._text], keep='first')

        # Use only a subset of the data
        df = df.sample(frac=frac).reset_index(drop=True)

        # Construct a unique label by concatenating main and sub-labels with the "|" character
        df[self._lbl] = df[self._main] + "|" + "|" + df[self._sub]

        # Filter out examples where the label occurs less than 50 times
        number_of_examples = df[self._lbl].value_counts().to_frame()
        df['is_bigger_than_50'] = df[self._lbl].isin(number_of_examples[number_of_examples[self._lbl] > 50].index)
        df['is_bigger_than_50'].value_counts()
        df = df[df['is_bigger_than_50'] == True]  # noqa E712

        return df

    def make_data_sets(self, df: pd.DataFrame, split: float = 0.9, columns: list[str] = None) \
            -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Create train and test datasets from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data.
        split : float, optional
            The fraction of the data to be used for testing. Default is 0.9, meaning 90% training and 10% testing.
        columns : list of str, optional
            The list of columns to be used for labels. If None, default columns [self._main, self._sub] are used.

        Returns
        -------
        tuple:
            A tuple containing train and test datasets:
            - texts : pd.Series
                The text data.
            - labels : pd.Series
                The labels.
            - train_texts : pd.Series
                The training text data.
            - train_labels : pd.Series
                The training labels.
            - test_texts : pd.Series
                The testing text data.
            - test_labels : pd.Series
                The testing labels.
        """
        # If columns are not provided, use default columns [self._main, self._sub]
        columns = columns or [self._main, self._sub]

        # Extract text and label columns from the DataFrame
        texts = df[self._text]
        labels = df[columns].apply('|'.join, axis=1)

        # Split the data into training and testing sets using stratified sampling
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=1-split,
            stratify=labels
        )

        return texts, labels, train_texts, train_labels, test_texts, test_labels

    def fit(self, train_texts: pd.Series, train_labels: pd.Series, optimized_parameters: bool = True) -> GridSearchCV:
        """
        Fit a logistic regression model using text data and labels.

        Parameters
        ----------
        train_texts : pd.Series
            The training text data.
        train_labels : pd.Series
            The training labels.
        optimized_parameters: bool
            Flag to switch between optimized (slower) or non optimized (faster) parameters.

        Returns
        -------
        GridSearchCV
            The fitted grid search model.
        """
        # Define a pipeline with text vectorization and logistic regression
        pipeline = Pipeline([
                ('vect', CountVectorizer(preprocessor=self.preprocessor, stop_words=self.stop_words)),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression()),
        ])

        if optimized_parameters:
            # Define hyperparameters for slow training with better optimization
            parameters = {
                    'clf__class_weight': (None, 'balanced', ),
                    'clf__max_iter': (300, 500, ),
                    'clf__penalty': ('l1', ),
                    'clf__multi_class': ('auto', ),
                    'clf__solver': ('liblinear', ),
                    'tfidf__norm': ('l2', ),
                    'tfidf__use_idf': (False, ),
                    'vect__max_df': (1.0, ),
                    'vect__max_features': (None, ),
                    'vect__ngram_range': ((1, 1), (1, 2), )
            }
        else:
            # Define hyperparameters for fast training with minimal optimization
            parameters = {
                    'clf__class_weight': (None, ),
                    'clf__max_iter': (300, ),
                    'clf__penalty': ('l1', ),
                    'clf__solver': ('liblinear', ),
                    'tfidf__norm': ('l2', ),
                    'tfidf__use_idf': (False, ),
                    'vect__max_df': (1.0, ),
                    'vect__max_features': (None, ),
                    'vect__ngram_range': ((1, 1), )
            }

        # Log the parameters
        logger.info('Performing grid search...')
        logger.info('Parameters used for grid search:')
        logger.info(json.dumps(parameters, indent=2))

        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            parameters,
            verbose=True,
            n_jobs=psutil.cpu_count(logical=False),
            cv=5
        )
        grid_search.fit(train_texts, train_labels)

        # Set the model attribute to the fitted grid search model
        self.model = grid_search

        logger.info('Grid search completed')
        return grid_search

    def validate_model(self, test_texts: pd.Series, test_labels: pd.Series,
                       dst_file: str | Path, dst_csv: str | Path, dst_validation: str | Path = None):
        """
        Validate the trained model using test data and generate evaluation metrics, a confusion matrix plot,
        and optional detailed validation results.

        Parameters
        ----------
        test_texts : pd.Series
            The testing text data.
        test_labels : pd.Series
            The true labels for the testing data.
        dst_file : str or Path
            The file path to save the confusion matrix plot.
        dst_csv : str or Path
            The file path to save the confusion matrix CSV.
        dst_validation : str, optional
            The file path to save detailed validation results (optional).
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        logger.info('Validating the generated model...')

        # Predict labels for the test data
        test_predict = self.model.predict(test_texts)

        # Calculate precision, recall, and accuracy scores
        precision = precision_score(test_labels, test_predict, average='macro', zero_division=0)
        recall = recall_score(test_labels, test_predict, average='macro')
        accuracy = accuracy_score(test_labels, test_predict)

        logger.info(f'Precision: {precision:.2f}')
        logger.info(f'Recall: {recall:.2f}')
        logger.info(f'Accuracy: {accuracy:.2f}')

        # Configure plot size for confusion matrix display
        plt.rcParams["figure.figsize"] = (30, 30)

        # Generate the confusion matrix plot
        disp = ConfusionMatrixDisplay.from_estimator(
            self.model,
            test_texts,
            test_labels,
            cmap=plt.cm.Blues,
            normalize=None,
            xticks_rotation='vertical'
        )

        # Save the confusion matrix plot as an image
        logger.info(f'Saving the confusion matrix plot as an image to "{dst_file}"')
        plt.savefig(dst_file)

        # Save the confusion matrix as a CSV file
        logger.info(f'Saving the confusion matrix plot as a CSV file to "{dst_csv}"')
        df2 = pd.DataFrame(disp.confusion_matrix, columns=disp.display_labels)
        df2.to_csv(dst_csv)

        # Save detailed validation results if specified
        if dst_validation:
            logger.info(f'Saving detailed validation results to "{dst_validation}"')
            with open(dst_validation, 'w') as csvfile:
                fieldnames = ['Text', 'predicted_category', 'actual_category']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                for _input, prediction, label in zip(test_texts, test_predict, test_labels):
                    if prediction != label:
                        writer.writerow(
                            {
                                'Text': re.sub(pattern="\\W", repl=" ", string=_input),
                                'predicted_category': prediction,
                                'actual_category': label
                            }
                        )

        logger.info('Validation completed')
