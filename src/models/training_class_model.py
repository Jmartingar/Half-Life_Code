import pandas as pd
import sys
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from classification_models import ClassificationModel

df_to_train = pd.read_csv(sys.argv[1])
output_path = sys.argv[2]
name_response = sys.argv[3]
random_seed = int(sys.argv[4])
preffix_export = sys.argv[5]

print("Processing random seed: ", random_seed)

responses = df_to_train[name_response].values
df_data = df_to_train.drop(columns=[name_response])

train_data, validation_data, train_response, validation_response = train_test_split(df_data, responses, random_state=random_seed, test_size=.20)

class_models = ClassificationModel(
    train_response=train_response,
    train_values=train_data,
    test_response=validation_response,
    test_values=validation_data
)

name_export = f"{output_path}{preffix_export}_exploring_{random_seed}.csv"
df_exploration = class_models.apply_exploring()
df_exploration.to_csv(name_export, index=False)