import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import random
import statistics
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math

#  read the dataset
df = pd.read_csv("df1111.csv")

df = df[df.Disease != 'Typhoid']

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)
disease_encoder = LabelEncoder()
symptoms_encoder = LabelEncoder()

X = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5']]
y = df['Disease'].values.tolist()
y = disease_encoder.fit_transform(y)

symptoms = np.unique(X)
symptom_ids = symptoms_encoder.fit_transform(symptoms)
symptom_map = dict(zip(symptoms_encoder.classes_, range(len(symptoms_encoder.classes_))))

for a in X.columns:
    X[a] = symptoms_encoder.transform(X[a])

y = pd.DataFrame(y, columns=["Disease"])
dataset = pd.merge(X, y, left_index=True, right_index=True)
dataset = dataset.values.tolist()


# split the dataset into test set and train set
def split_dataset(dataset, ratio):
    train_size = int(len(dataset) * ratio)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return [train_set, test_set]


# split the dataset by class values & return a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)

    for keys, values in separated.copy().items():
        if len(values) <= 1:
            del (separated[keys])
    return separated


# calculate mean, standard deviation and count for each column in dataset
def summarize_dataset(dataset):
    summaries = [(statistics.mean(column), statistics.stdev(column, ), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


# split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# calculate the gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# calculate the probabilities of predicting each class for a given data
def calculate_class_probabilities(summaries, input):
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, standard_deviation, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(input[i], mean, standard_deviation)
    return probabilities


# predict the class for a given data
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def get_predictions(summary, test_data):
    predictions = []
    for i in range(len(test_data)):
        predictions.append(predict(

            summary, test_data[i]))
    return predictions


def get_accuracy(test_data, predictions):
    correct = 0
    for x in range(len(test_data)):
        if test_data[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_data))) * 100


# split dataset into test set and train set
split_ratio = 0.85
train_set, test_set = split_dataset(dataset, split_ratio)
print("Split {0} rows into train :- {1} and test :- {2} rows.".format(len(dataset), len(train_set), len(test_set)))

# prepare model
summaries = summarize_by_class(dataset)

# test the prepared model
prediction = get_predictions(summaries, test_set)

# get the accuracy
accuracy = get_accuracy(test_set, prediction)
print("Accuracy :- {0}%".format(accuracy))


# user_data = ['headache', 'sweating', 'cough', 'high_fever', 'breathlessness']
# user_data = symptoms_encoder.transform(user_data)
# user_data = np.reshape(user_data, (1, 5))
# predict_user_data = get_predictions(summaries, user_data)
# print(disease_encoder.inverse_transform(predict_user_data))

# def predict_from_user_data(user_data):
#     user_data = symptoms_encoder.transform(user_data)
#     user_data = np.reshape(user_data, (1, 5))
#     predict_user_data = get_predictions(summaries, user_data)
#     return disease_encoder.inverse_transform(predict_user_data)
#
#
# print(predict_from_user_data(['headache', 'sweating', 'cough', 'high_fever', 'breathlessness']))

app = FastAPI()


class Request_body(BaseModel):
    """
    Request body for the API.
    """
    symptom1: str
    symptom2: str
    symptom3: str
    symptom4: str
    symptom5: str


@app.post("/api/v1/predict")
def predict(request_body: Request_body):
    """
    Predict the probability of having a disease
    """
    user_input = [request_body.symptom1, request_body.symptom2, request_body.symptom3,
                  request_body.symptom4, request_body.symptom5]

    user_input = symptoms_encoder.transform(user_input)
    user_input = np.reshape(user_input, (1, 5))

    predict_user_input = get_predictions(summaries, user_input)
    return disease_encoder.inverse_transform(predict_user_input)
