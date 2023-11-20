import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from settings import LOG_LEVEL, MODELS_DIRECTORY, SIGNALS_CATEGORY_URL
from sklearn.model_selection import GridSearchCV

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)-4s %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(level=LOG_LEVEL)
logger.addHandler(handler)


def load_pickle(file_path: str | Path) -> Any:
    """
    Load a pickled file from the given file path.

    Note: If the file does not exist, an error is logged, and None is returned.

    Parameters
    ----------
    file_path : str
        The path to the pickled file.

    Returns
    -------
    object : any
        The unpickled object loaded from the file.
    """
    if not os.path.exists(file_path):
        logger.error(f'File does not exists "{file_path}"')
    else:
        # Load and return the pickled object
        return joblib.load(file_path)


def classify_text(text: str, model: GridSearchCV, categories: list[str], top_n: int = 100) \
        -> tuple[list[str], list[float]]:
    """
    Get the top categories and their associated probabilities for a given text.

    Parameters
    ----------
    text : str
        The input text for which categories and probabilities are to be predicted.
    model : GridSearchCV
        A trained classifier model with a `predict_proba` method.
    categories : list
        A list of category labels.
    top_n : int, optional
        Number of top categories to retrieve, default is 100.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - List of URLs for the top categories.
        - List of probabilities sorted in descending order.
    """
    # Predict probabilities for the given text
    proba_predictions = model.predict_proba([text, ])

    # Get probabilities sorted in descending order
    sorted_probs = list(reversed(sorted(proba_predictions[0])))[:top_n]

    # Get the top indices corresponding to the top_n categories
    top_indices = np.argsort(proba_predictions)[0][-top_n:][::-1]

    # Create URLs for the top categories
    top_category_urls = [f'{SIGNALS_CATEGORY_URL}{categories[z]}' for z in top_indices]

    return top_category_urls, sorted_probs


# Load the main model and categories from pickled files
main_model = load_pickle(file_path=os.path.join(MODELS_DIRECTORY, 'main_model.pkl'))
main_categories = load_pickle(file_path=os.path.join(MODELS_DIRECTORY, 'main_slugs.pkl'))

# Create a partial function for classifying text using the main model
main_model_classify_text = partial(classify_text, model=main_model, categories=main_categories)

# Load the sub-model and sub-categories from pickled files
sub_model = load_pickle(file_path=os.path.join(MODELS_DIRECTORY, 'sub_model.pkl'))
sub_categories = load_pickle(file_path=os.path.join(MODELS_DIRECTORY, 'sub_slugs.pkl'))

# Create a partial function for classifying text using the sub-model
sub_model_classify_text = partial(classify_text, model=sub_model, categories=sub_categories)
