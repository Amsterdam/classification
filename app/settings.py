import os

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
SIGNALS_CATEGORY_URL = os.getenv('SIGNALS_CATEGORY_URL', 'https://backend.signalen.demoground.nl/signals/v1/public/terms')
MODELS_DIRECTORY = os.getenv('MODELS_DIRECTORY', '/app/models')
