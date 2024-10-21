# Multiclass Iris Species Classification Using PyTorch
## Objective
The objective of this task is to build a multiclass classification model using PyTorch to classify three Iris species (setosa, versicolor, and virginica) based on four flower features (sepal length, sepal width, petal length, petal width). The model is expected to achieve an accuracy of over 95% and be evaluated using key metrics such as accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC curves.

## Dataset Description
We are using the well-known Iris dataset, which contains 150 samples of Iris flowers with four features and three species as target labels.

**Features:**

1- Sepal length (in cm)

2- Sepal width (in cm)

3- Petal length (in cm)

4- Petal width (in cm)

**Target Classes (Species):**

1- Iris-setosa

2- Iris-versicolor

3- Iris-virginica

The dataset is pre-processed by normalizing the feature values and one-hot encoding the target labels for classification. The data is then split into training (80%), validation (10%), and testing (10%) sets.

## Steps to Run the Code in Jupyter
**1- Install the Required Dependencies** 

(see the dependencies section below).

**2- Download the Dataset:**

Ensure that you have the Iris dataset in a CSV format at the specified path in the code, for example:

'D:/noody/Deep learning/Lab 2/iris.csv'

**3- Open Jupyter Notebook:** 

You can launch Jupyter Notebook by running:

jupyter notebook

**4- Load the Project:**

Place the code in a Jupyter notebook cell.

Make sure the dataset path is correctly specified.

**5- Run the Notebook:**

Run all the cells sequentially, starting from importing dependencies to training the model, evaluating its performance, and visualizing the results (confusion matrix, ROC-AUC curves, etc.).

**6- Training and Evaluation:**

The training loop will run for 100 epochs, and after each epoch, the training and validation losses will be printed.

At the end of the notebook, the test dataset will be used to evaluate the model, and accuracy, confusion matrix, and classification report will be displayed.

The ROC-AUC curves for each class will be plotted for better performance insights.

## Dependencies and Installation Instructions
****Required Dependencies****

The project relies on the following libraries:

**1- PyTorch:**

For building and training the neural network.

**2- Sklearn:** 

For dataset manipulation, preprocessing, and evaluation metrics.

**3- Pandas:** 

For handling the dataset.

**4- Numpy:** 

For numerical operations.

**5- Matplotlib:** 

For plotting graphs (training/validation loss and ROC-AUC curves).

**6- Seaborn:** 

For enhanced visualizations (used for confusion matrix).

****Installation Instructions****

To install all the necessary packages, follow the steps below:

**1- Install Python 3.x:**

You need to have Python 3.x installed on your machine. You can download Python from the official site: https://www.python.org/downloads/

**2- Install PyTorch:** 

You can install PyTorch using the following command (CPU version):

pip install torch torchvision torchaudio

**3- Install Scikit-learn:**

Use the following command to install Scikit-learn for data manipulation and evaluation metrics:

pip install scikit-learn

**4- Install Pandas and Numpy:**

To install pandas and numpy, run:

pip install pandas numpy

**5- Install Matplotlib and Seaborn:** 

These libraries are used for plotting:

pip install matplotlib seaborn

**6- Install Jupyter Notebook:**

If you don’t already have Jupyter Notebook installed, you can install it using:

pip install notebook

****Additional Notes****

1- If you are using a GPU, you can install the GPU version of PyTorch for faster training.

2- Ensure that the path to the dataset is correct when running the notebook (D:/noody/Deep learning/Lab 2/iris.csv).

3- The model will train over 100 epochs, and you can modify the hyperparameters in the code if needed.

## Evaluation Metrics
After training, the following metrics will be used to evaluate the model's performance:

**1- Accuracy Score:**

Overall percentage of correctly classified samples.

**2- Confusion Matrix:**

Shows the true positives, false positives, true negatives, and false negatives for each class.

**3- Classification Report:**

Includes precision, recall, F1-score for each class.

**4- ROC-AUC Curve:**

Plots the ROC curve for each class and calculates the AUC score, providing insights into how well the model separates the classes.