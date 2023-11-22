import pandas as pd
import psutil
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def get_pipeline(stop_words: str = None) -> Pipeline:
    """
    Parameters
    ----------
    stop_words : str or None
        A string or None indicating the list of stop words to be used for
        the CountVectorizer.

    Returns
    -------
    pipeline : Pipeline
        A scikit-learn Pipeline object that consists of the following steps:
        1. CountVectorizer - Converts a collection of text documents to a
           matrix of token counts with an option to remove stop words based on
           the provided stop_words parameter.
        2. TfidfTransformer - Transforms the matrix produced by CountVectorizer
           into a normalized tf-idf representation.
        3. LogisticRegression - Implements the logistic regression algorithm
           for classification.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words=stop_words)),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression()),
    ])
    return pipeline


def get_parameters(optimized: bool = True) -> dict:
    """
    Parameters
    ----------
    optimized : bool, optional
        Flag to determine whether to return the optimized parameters or the
        base parameters.
        If True (default), the optimized parameters will be returned.
        If False, the base parameters will be returned.

    Returns
    -------
    dict
        A dictionary containing the parameters for GridSearchCV.
    """
    base_parameters = {
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

    if optimized:
        return {**base_parameters, **{
            'clf__class_weight': (None, 'balanced', ),
            'clf__max_iter': (300, 500, ),
            'clf__multi_class': ('auto', ),
            'vect__ngram_range': ((1, 1), (1, 2), )
        }}
    else:
        return base_parameters


def perform_grid_search_cv(train_texts: pd.Series, train_labels: pd.Series,
                           pipeline: Pipeline, parameters: dict = None) -> GridSearchCV:
    """
    Performs grid search cross-validation for a given pipeline and parameters.

    Parameters
    ----------
    train_texts : pd.Series
        The training texts.

    train_labels : pd.Series
        The corresponding training labels.

    pipeline : Pipeline
        The pipeline object containing the feature extraction and model.

    parameters : dict, optional
        The parameter grid for grid search. Default is None.

    Returns
    -------
    grid_search : GridSearchCV
        The grid search object with the best estimator and hyperparameters.

    """
    if parameters is None:
        parameters = get_parameters(optimized=True)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        n_jobs=psutil.cpu_count(logical=False),
        cv=5
    )
    grid_search.fit(train_texts, train_labels)
    return grid_search


def generate_category_paths(model: GridSearchCV, columns: list[str]) \
        -> list[str]:
    """
    Generates URL paths for categories based on model classes.

    If `columns` has more than one column, it is assumed that model.classes_
    contains strings with format 'category|subcategory'.
    In this case, URL paths with both categories and subcategories will be
    generated.

    If `columns` has only one column, then the classes are treated as
    categories only.

    Parameters
    ----------
    model : object
        A trained model object with 'classes_' attribute.
    columns : list of str
        A list of string specifying column names in data.

    Returns
    -------
    paths : list of str
        A list of URL paths as strings.

    """
    if len(columns) > 1:
        categories = [
            x.split("|")
            for x in model.classes_
        ]

        paths = [
            f"/categories/{category[0]}/sub_categories/{category[1]}"
            for category in categories
        ]
    else:
        paths = [
            f"/categories/{category}"
            for category in model.classes_
        ]

    return paths
