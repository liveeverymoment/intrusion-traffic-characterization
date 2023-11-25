# intrusion-traffic-characterization
## The codebase pertains to machine learning model creation for the purpose of intrusion traffic characterization.
### The dataset used for training the machine learning model is CIC-IDS2017.

Steps performed for training model:
1. Preprocess of dataset
    1. Separate y variable (attacks label).
    2. Replace infinite values with nan.
    3. Remove all rows with nan values.
    4. Perform Integer (LabelEncoding) encoding of y variable.
    5. (conditional) If the y variable is more then, perform one hot encoding.
    6. Normalize numerical columns (attributes).
    7. (optional) Perform feature selection if needed.
3. Create train, test, and evaluate the dataset
    - Use the 60:30:10 ratio to divide data for training, testing, and evaluation.
4. Apply machine learning algorithms
    - Calculate the time to train.
5. Test testing data with the ML model
    - Calculate: Accuracy, F1, Precision, Recall, Confusion matrix
6. Evaluate with the evaluation dataset.
7. Vary algorithm's variables (attribute values) to get the optimum result.
8. Document, and compare with standard research results.

Bye, happy coding!
