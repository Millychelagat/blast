import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle
from sklearn.preprocessing import LabelEncoder


def create_model(df):
    X = df.drop("Disease", axis=1)
    Y = df['Disease']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_test, X_train, y_test, y_train = train_test_split(X, Y, test_size=0.2, random_state=12)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f'Accuracy: {accuracy}')

    return lr, scaler


def get_clean_data():
    df = pd.read_csv("../data/animal_disease_dataset.csv")
    condition = (df['Animal'] == "cow")
    df = df[condition]

    col_to_encode = ['Animal', 'Disease']

    df, encoders = label_encode_columns(df, col_to_encode)
    df = df.drop("Animal", axis=1)
    new_features = ['blisters on gums', 'blisters on hooves', 'blisters on mouth',
                    'blisters on tongue', 'chest discomfort', 'chills', 'crackling sound',
                    'depression', 'difficulty walking', 'fatigue', 'lameness', 'loss of appetite',
                    'painless lumps', 'shortness of breath', 'sores on gums', 'sores on hooves',
                    'sores on mouth', 'sores on tongue', 'sweats', 'swelling in abdomen',
                    'swelling in extremities', 'swelling in limb', 'swelling in muscle',
                    'swelling in neck']

    for feature in new_features:
        df[feature] = 0

    for index, row in df.iterrows():
        for symptom_column in ['Symptom 1', 'Symptom 2', 'Symptom 3']:
            symptom = row[symptom_column]
            if symptom in new_features:
                df.loc[index, symptom] = 1
    # Remove redundant colums
    df.drop(['Symptom 1', 'Symptom 2', 'Symptom 3'], axis=1, inplace=True)

    # Make diseae the last column
    cols = list(df.columns)
    cols.remove('Disease')
    cols.append('Disease')
    df = df[cols]
    return df


def label_encode_columns(data, columns):
    encoders = {}
    lb = LabelEncoder()
    for column in columns:
        data[column] = lb.fit_transform(data[column])
        encoders[column] = lb
    return data, encoders


def main():
    df = get_clean_data()
    model, scaler = create_model(df)
    with open("cattle_diseases_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("cattle_diseases_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
