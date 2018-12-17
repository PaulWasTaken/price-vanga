import pandas as pd

from datetime import datetime
from os import path
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor

TRAIN_DATA_PATH = "train_data.csv"
TEST_DATA_PATH = "test_data.csv"


def date_converter(row):
    return (datetime.strptime(row["Дата заезда"], "%Y-%m-%d") - datetime.strptime(row["Дата создания"],
                                                                                  "%Y-%m-%d")).days


def extract_x_and_y(data: pd.DataFrame):
    dependent_variable = data.filter(["Стоимость тарифа"], axis=1)
    arguments = data[["Глубина бронирования", "До заезда", "Сезон", "Выходной"]]
    return arguments, dependent_variable


def extract_season(row):
    month = datetime.strptime(row["Дата создания"], "%Y-%m-%d").month
    if month in (12, 1, 2,):
        return 1
    elif month in (3, 4, 5):
        return 2
    elif month in (6, 7, 8):
        return 3
    elif month in (9, 10, 11):
        return 4
    else:
        raise ValueError


if not (path.isfile(TRAIN_DATA_PATH) and path.isfile(TEST_DATA_PATH)):
    from extender import extend_data

    extend_data()

    data = pd.read_csv("bookings_example.csv", encoding="utf-8")
    data["До заезда"] = data.apply(lambda row: date_converter(row), axis=1)
    data["Сезон"] = data.apply(lambda row: extract_season(row), axis=1)

    data[data["Дата создания"] < "2018-01-01"].to_csv(TRAIN_DATA_PATH, index=False)
    data[data["Дата создания"] >= "2018-01-01"].to_csv(TEST_DATA_PATH, index=False)

train_data = pd.read_csv(TRAIN_DATA_PATH)
test_data = pd.read_csv(TEST_DATA_PATH)

x_train, y_train = extract_x_and_y(train_data)
x_test, y_test = extract_x_and_y(test_data)

model = XGBRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
