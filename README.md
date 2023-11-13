# Machine learning tool
This is a service used by the [signals application](https://github.com/Amsterdam/signals) and provides the ability to
predict the category that a signal belongs to. It achieves this using existing data, either from an old system or from
signals itself. See the "Input data" section below for more information on the format that is required.

The model is based on [sklearn](https://scikit-learn.org) and is trained by removing things like "stop words" and
special characters, and subsequently counting the stemmed version of the remaining words. The outcome of that processed
is then transformed to a [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) format. Finally, to form a statistic
model, regression analysis is performed in the form of
[logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)

To get a prediction from the model there is an API, built using Flask. See section "Running service" below for more
information.

# Building the Docker images
Navigate to the root directory and pull the  relevant images and build the services:

```console
docker-compose build
```

# Input data

The `CSV` input file must have at least the following columns:

| column      | description        |
|-------------|--------------------|
| Text        | message            |
| Main        | Main category name |
| Sub         | Sub category name  |

The columns must be in the order `Text,Main,Sub`, no header is required.

# Training model using docker compose
To train the model run the following command:

```
docker-compose run --rm train python train.py --csv=/input/{name of csv file} --columns={name of column}
```

for example:

```
docker-compose run --rm train python train.py --csv=/input/dump.csv --columns=Main
```

This will produce a set of files pickled using [joblib](https://joblib.readthedocs.io) and some files that can be used
to verify the accuracy of the model in the form of a
[confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).  
The files will be saved in the `ouput` directory.

In the example above this would result in:
- `/output/Main_model.pkl`
- `/output/Main_slugs.pkl`
- `/output/Main_dl.csv`
- `/output/Main_matrix.csv`
- `/output/Main_matrix.pdf`

# Running service
The service is a standalone API built on the [Flask](https://flask.palletsprojects.com) framework. In order for it to
be able to use the model that was trained the pickle files are required.
Copy the pickle files listed below into the `/models` directory or the directory you have configured through the
`MODELS_DIRECTORY` environmental variable.

| file           | description             |
|----------------|-------------------------|
| main_model.pkl | model for main category |
| sub_model.pkl  | model for sub category  |
| main_slugs.pkl | slugs for main category |
| sub_slugs.pkl  | slugs for sub category  |

In order for the API to produce useful results for the signals application, it is important to provide a base url for
the backend portion of the application. This can be achieved by setting the environmental variable
`SIGNALS_CATEGORY_URL`.

To activate the flask api run:
```shell
docker-compose up -d web
```

To test the service using your browser, open http://localhost:8140/web_pages/index.html.
Typically, a POST request with a body similar to:
```json
{
  "text": "afval"
}
```
should be made to http://localhost:8140/signals_mltool/predict, to get a prediction.
This should give a response with a body similar:
```json
{
  "hoofdrubriek": [
    [
      "http://localhost:8000/signals/v1/public/terms/categories/afval"
    ],
    [
      0.7629584838555712
    ]
  ],
  "subrubriek": [
    [
      "http://localhost:8000/signals/v1/public/terms/categories/afval/sub_categories/huisafval"
    ],
    [
      0.56709391826473
    ]
  ]
}
```
As can be seen in the example response, a full url is given, while for the training process only category names were
provided. This happens because before the model is trained the category names are converted to slugs in the same way
the signals application converts them. Subsequently, the url is constructed using the base url in the same way that the
signals application constructs those urls.
