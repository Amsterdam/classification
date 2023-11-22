import re

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


def preprocess_text(text: str, stemmer: SnowballStemmer = None) -> str:
    """
    Preprocess the input text by converting it to lowercase,
    removing special characters, and applying stemming to each word.

    Parameters
    ----------
    text : str
        The input text to be preprocessed.
    stemmer : nltk.stem.SnowballStemmer, optional
        The stemmer to be used for stemming words. If not provided,
        SnowballStemmer with the `Dutch` language will be used by default.

    Returns
    -------
    str
        The preprocessed text.
    """
    # Convert the text to lowercase
    text = text.lower()

    # Remove special characters using regular expression
    text = re.sub(pattern=r"\W", repl=" ", string=text)

    # Tokenize the text
    token_words = word_tokenize(text)

    # Use the provided stemmer or default to DutchStemmer if none provided
    if stemmer is None:
        stemmer = SnowballStemmer(language="dutch", ignore_stopwords=True)

    # Apply stemming to each word using the provided stemmer
    # and join the processed words back into a single string
    return " ".join([
        stemmer.stem(word=word)
        for word in token_words
    ])


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a DataFrame, removing empty entries
    and duplicates.

    Parameters
    ----------
    filepath : str
        The path to the CSV file to load

    Returns
    -------
    pd.DataFrame
        The loaded and cleaned DataFrame.
    """
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Rename columns for clarity
    df.columns = ["main", "sub", "text"]

    # Drop rows with empty values (None or NaN) in specified columns
    df = df.dropna(
        axis=0,
        how="any",
        subset=["main", "sub", "text"],
        inplace=False,
    )

    # Drop duplicate texts (keep the first occurrence)
    df = df.drop_duplicates(subset=["text"], keep="first")

    return df


def sample_and_create_label(df: pd.DataFrame, frac: float = 1.0) \
        -> pd.DataFrame:
    """
    Take a subset of the data and create a new label by concatenating
    'main' and 'sub' columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    frac : float, optional
        The fraction of the DataFrame to return.

    Returns
    -------
    pd.DataFrame
        The DataFrame after sampling and label creation.
    """
    df = df.sample(frac=frac).reset_index(drop=True)
    df["label"] = df["main"] + "|" + df["sub"]

    return df


def filter_labels(df: pd.DataFrame, min_frequency: int = 50) -> pd.DataFrame:
    """
    Filter the DataFrame to only include rows where the label occurs at least
    min_frequency times.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    min_frequency : int, optional
        The minimum allowed frequency of each label. Default is 50.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """
    label_counts = df["label"].value_counts()
    frequent_labels = label_counts[label_counts > min_frequency].index

    return df[df["label"].isin(frequent_labels)]


def generate_training_testing_data(df: pd.DataFrame, split: float = 0.9,
                                   columns: list[str] = None) \
        -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Create train and test datasets from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    split : float, optional
        The fraction of the data to be used for testing. Default is 0.9,
        meaning 90% training and 10% testing.
    columns : list of str, optional
        The list of columns to be used for labels. If None, default columns
        ['main', 'sub'] are used.

    Returns
    -------
    tuple:
        A tuple containing train and test datasets:
        - train_texts : pd.Series
            The training text data.
        - train_labels : pd.Series
            The training labels.
        - test_texts : pd.Series
            The testing text data.
        - test_labels : pd.Series
            The testing labels.
    """
    # If columns are not provided, use default columns ['main', 'sub']
    columns = columns or ["main", "sub"]

    # Extract text and label columns from the DataFrame
    texts = df["text"]
    labels = df[columns].apply("|".join, axis=1)

    # Split the data into training and testing sets using stratified sampling
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=1 - split,
        stratify=labels
    )

    return train_texts, train_labels, test_texts, test_labels
