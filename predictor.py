import pandas as pd

from datetime import datetime, timedelta
from math import e, sqrt
from os import path
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor

TRAIN_DATA_PATH = "train_data.csv"
TEST_DATA_PATH = "test_data.csv"
DATA_FILE = "original_bookings_example.csv"


class RussianHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1), Holiday("New Year", month=1, day=2), Holiday("New Year", month=1, day=3),
        Holiday("New Year", month=1, day=4), Holiday("New Year", month=1, day=5), Holiday("New Year", month=1, day=6),
        Holiday("New Year", month=1, day=7),
        Holiday("Men's Day", month=2, day=23),
        Holiday("Women's Day", month=3, day=8),
    ]


def date_converter(row):
    return (datetime.strptime(row["Дата заезда"], "%Y-%m-%d") - datetime.strptime(row["Дата создания"], "%Y-%m-%d")).days


FEATURES_NAME = ["Глубина бронирования", "До заезда", "Сезон",
                 "День недели", "Праздник", "Дней_до/Глубина",
                 "День_недели/Сезон"]


def extract_x_and_y(data: pd.DataFrame):
    dependent_variable = data.filter(["Стоимость тарифа"], axis=1)
    arguments = data[FEATURES_NAME]
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


def extract_weekday(row):
    return datetime.strptime(row["Дата создания"], "%Y-%m-%d").weekday() + 1


CALENDAR = RussianHolidayCalendar()
HOLIDAYS = CALENDAR.holidays(start='2014-01-01', end='2018-12-31').to_pydatetime()
HOLIDAYS_WITH_SHIFTS = set()
for holiday in HOLIDAYS:
    HOLIDAYS_WITH_SHIFTS |= {str(date.date()) for date in pd.date_range(holiday - timedelta(8), holiday + timedelta(1)).to_pydatetime()}


def mark_if_holiday(row):
    return 2 if row["Дата заезда"] in HOLIDAYS_WITH_SHIFTS else 1


def bind_weekend_season(row):
    return round(e ** (row["Сезон"] + row["День недели"]), 2)


def bind_before_deep(row):
    return round(sqrt(row["До заезда"] * row["Глубина бронирования"]), 2)


FORCE_REGENERATE = True
if (not (path.isfile(TRAIN_DATA_PATH) and path.isfile(TEST_DATA_PATH))) or FORCE_REGENERATE:
    data = pd.read_csv(DATA_FILE, encoding="utf-8")
    data["До заезда"] = data.apply(lambda row: date_converter(row), axis=1)
    data["Сезон"] = data.apply(lambda row: extract_season(row), axis=1)
    data["День недели"] = data.apply(lambda row: extract_weekday(row), axis=1)
    data["Праздник"] = data.apply(lambda row: mark_if_holiday(row), axis=1)

    # Add correlated features
    data["День_недели/Сезон"] = data.apply(lambda row: bind_weekend_season(row), axis=1)
    data["Дней_до/Глубина"] = data.apply(lambda row: bind_before_deep(row), axis=1)

    data[data["Дата создания"] < "2018-01-01"].to_csv(TRAIN_DATA_PATH, index=False)
    data[data["Дата создания"] >= "2018-01-01"].to_csv(TEST_DATA_PATH, index=False)


train_data = pd.read_csv(TRAIN_DATA_PATH)
test_data = pd.read_csv(TEST_DATA_PATH)

x_train, y_train = extract_x_and_y(train_data)
x_test, y_test = extract_x_and_y(test_data)

model = XGBRegressor()
model.fit(x_train, y_train)

for name, imp in zip(FEATURES_NAME, model.feature_importances_):
    print("Name: %s; Importance: %.10f" % (name, imp))

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

print(accuracy_score(y_test, predictions))
print(model.score(x_test, y_test))
print(sqrt(mean_squared_error(y_test, predictions)))
