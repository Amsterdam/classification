# Training the models using Azure ML
Training the models can take a very long time depending on the resources you have available on your machine.  
It may be worth it to use an external service like [Azure ML](https://ml.azure.com/) to train them.

In order to do just that this directory contains a pipeline definition and a script to make the process a bit easier.

## Setting up
To be able to do anything with Azure ML, it will need to be setup. That will require doing a few things:
- Signup for an Azure account
- Have access to an active subscription
- Create a workspace and compute resource

A basic description on how to create a workspace and compute resource can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources).

In order to be able to run the pipeline the `az` cli needs to be installed, there are some decent instructions on how
to do that [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2).

The most important thing to do after installing the cli and extension is to run:
```shell
az login
```

There is no need for any further configuration of the cli in order to use the script that is provided in this directory.
What is required, is that the `.env.example` file needs to be copied to `.env` and then be modified to contain the right
values.

| name         | description                                                                                                         |
|--------------|---------------------------------------------------------------------------------------------------------------------|
| SUBSCRIPTION | The Azure subscription id you would like to use.                                                                    |
| GROUP        | The id of the resource group within the subscription.                                                               |
| WORKSPACE    | The id of the Azure ML workspace.                                                                                   |
| COMPUTE      | The name of the compute resource within the workspace, prefixed with `azureml:`, for example `azureml:cpu_compute`. |

## How to train the models
The pipeline will use the training container image that is published to Docker hub to perform the actual training.
It is very important that the version of the training container matches the version of the classification web service.
So make sure to use the latest version of both to be certain.

To kick off a pipeline, simply run:
```shell
./run.sh path/to/file.csv
```
after that a new tab should automatically open in the browser, taking you to the jobs.

Once they're both completed the files should be available in the workspace storage and can be downloaded there.
The easiest way to locate them is to open the job details and to click on the "Data asset" link in the "Outputs"
section.  
That will take you to page with a section called "Data sources", which in turn has a listing of "Actions" that allow you
to see the actual files.
