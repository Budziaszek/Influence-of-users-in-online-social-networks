from statistics import mean, stdev


def without_zeros(value):
    if value == 0:
        return False
    return True


def without_none(value):
    if value is None:
        return False
    return True


def coefficient_of_variation(data):
    if len(data) > 1 and mean(data) > 0:
        return stdev(data) / mean(data)
    else:
        return 0


def make_data_positive(data):
    """
    Adds absolute value of data minimum to each element in order to make data strictly positive.
    :param data: array
        DataProcessing to transform
    :return: array
        Transformed data
    """
    minimum = min(data)
    return data + abs(minimum) + 1


def modify_data(data, data_condition_function, new_value=0):
    """
    Removes unwanted values from data array.
    :param data: array
        DataProcessing that will be modified
    :param data_condition_function:
        Function which defines if data should stay in array or be removed
    :return: array
        Modified array
    """
    data_modified = []
    for d in data:
        if d is not None and (data_condition_function is None or data_condition_function(d)):
            data_modified.append(d)
        else:
            data_modified.append(new_value)
    return data_modified
