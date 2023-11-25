# intrusion-traffic-characterization
The codebase pertains to machine learning model creation for the purpose of intrusion traffic characterization.
The dataset used for training the machine learning model is CIC-IDS2017.

Steps performed for training model:
1. Preprocess of dataset
    1.1 Separate y variable (attacks label).
    1.2 Replace infinite values to nan.
    1.3 Remove all rows with nan values.
    1.4 Perform Integer (LabelEncoding) encoding of y variable.
    1.5 (conditional) If the y variable is more then, perform one hot encoding.
    1.6 Normalize numerical columns (attributes).
    1.7 (optional) Perform feature selection if needed.
2. Create train, test, and evaluate the dataset
    2.1 Use the 60:30:10 ratio to divide data for training, testing, and evaluation.
3. Apply machine learning algorithms
    3.1 Calculate the time to train.
4. Test testing data with the ML model
    4.1 Calculate: Accuracy, F1, Precision, Recall, Confusion matrix
5. Evaluate with evaluation dataset.
6. Vary algorithm's variables (attribute values) to get the optimum result.
7. Document, and compare with standard research results.

Bye, happy coding!
