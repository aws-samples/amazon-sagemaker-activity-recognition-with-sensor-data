# Activity Recognition Using SmartPhone Sensor Data on Amazon SageMaker

Activity recognition refers to the process of identifying an activity from raw activity input data using machine learning. This data can come from wearable sensors, such as watches and accelerometers, or from cameras.

In this notebook, we will demonstrate how to train and deploy an activity recognition model (LSTM) on [Amazon SageMaker](https://aws.amazon.com/sagemaker/) using data from smartphone sensors.

For this problem, we will use a dataset from the [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones). The data includes 6 different activities:
- walking
- laying
- sitting
- standing
- walking upstairs
- walking downstairs

In the dataset, there are 3 main signal types in the raw data:
- total acceleration
- body acceleration
- body gyroscope

Each signal type has 3 axises of data, meaning there are 9 total variables for each time step.

The data has already been processed with the following configuration:
- Overlapping windows of 2.56 seconds (128 time steps)
- One 'row' (window) has 128 * 9 = 1152 features
- Features stored in separate files in 'Inertial Signals' subdirectories (9 input files to read)
- Labels stored in separate file (1 output file to read)

All you need to reproduce the example is ```sagemaker_torch.ipynb``` and ```requirements.txt```. This notebook will generate all of the necessary training scripts and data files. The notebook was tested on a ml.t3.medium (2 vCPU and 4 GB memory) using the Python3 kernel and the Data Science SageMaker Studio image. Be sure to use a SageMaker Studio user created after December 2022 to avoid the bug described [here](https://github.com/aws/amazon-sagemaker-examples/issues/3713).

## Prerequisites:
- AWS account
- Valid IAM credentials (can be provided through SageMaker Notebook instance or SageMaker Studio notebook) with access to SageMaker training and deployment jobs

## License
This library is licensed under the MIT-0 License. See the LICENSE file.
