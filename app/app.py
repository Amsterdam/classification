from flask import Flask, request
from flask_cors import CORS

from classifier import main_model_classify_text, sub_model_classify_text
from settings import TOP_N_PREDICTIONS

application = Flask(__name__)
CORS(application)


@application.get('/health')
def pong():
    """
    Check to see the health of the application, confirming if it is up and running
    """
    return {
        'health': 'awesome'
    }


@application.post('/signals_mltool/predict')
def predict():
    """
    Endpoint for predicting main and sub categories based on input text.

    This endpoint takes an input text from the request body and predicts the main and sub categories. The top X
    categories along with their probabilities are returned in the response.
    """
    main_category_urls, main_probs = main_model_classify_text(text=request.json['text'], top_n=TOP_N_PREDICTIONS)
    sub_category_urls, sub_probs = sub_model_classify_text(text=request.json['text'], top_n=TOP_N_PREDICTIONS)

    return {
        'hoofdrubriek': [
            main_category_urls,
            main_probs,
        ],
        'subrubriek': [
            sub_category_urls,
            sub_probs,
        ],
    }


if __name__ == '__main__':
    application.run(port=8000)
