from datetime import date

DATA_PATH = "bookings_example.csv"


def extract_data(s):
    numbers = s.split('-')
    numbers = map(lambda x: int(x), numbers)
    return date(*numbers)


def get_first_arrival(lines):
    first_arrival = date.max
    for line in lines:
        current_arrival = extract_data(line[3])
        if current_arrival < first_arrival:
            first_arrival = current_arrival
    return first_arrival


def get_data_lines():
    with open(DATA_PATH, "r", encoding="utf-8") as history:
        lines = history.readlines()
        is_first_line = True
        for line in lines:
            if is_first_line:
                is_first_line = False
                continue
            yield line[:-1].split(",")


def save_data_lines(lines):
    with open(DATA_PATH, "w", encoding="utf-8") as extended:
        extended.write(
            "Стоимость тарифа,Дата создания,Глубина бронирования,Дата заезда,День,Количество бронирований\n")
        str_lines = list(map(lambda x: ",".join(map(lambda l: str(l), x)) + "\n", lines))
        extended.writelines(str_lines)


def extend_lines_with_days(lines):
    first_arrival = get_first_arrival(lines)
    for line in lines:
        current_arrival = extract_data(line[3])
        days = (current_arrival - first_arrival).days + 1
        line.append(days)


def extend_lines_with_booking_count(lines):
    booking_history = {}
    for line in lines:
        arrival_date = line[3]
        if arrival_date in booking_history:
            booking_history[arrival_date] += 1
        else:
            booking_history[arrival_date] = 1
        line.append(booking_history[arrival_date])


lines = list(get_data_lines())
extend_lines_with_booking_count(lines)
extend_lines_with_days(lines)
save_data_lines(lines)
pass
