import numpy as np
import statistics
import logging


class Histogram:
    def __init__(self, x_scale, size_scale=None):
        self.x_scale = [round(x, 15) for x in x_scale]
        if size_scale is not None:
            self.size_scale = [round(s, 15) for s in size_scale]
            self.size_labels = self.get_labels(self.size_scale)
        else:
            self.size_scale = ["all"]
            self.size_labels = ["all"]
        self.array = np.zeros([len(size_scale) - 1, len(self.x_scale) - 1])
        self.x_labels = self.get_labels(self.x_scale)

    @staticmethod
    def check_index(value, scale_array):
        if value == scale_array[0]:
            return 0
        if value == scale_array[-1]:
            return len(scale_array)-2
        else:
            for i in range(1, len(scale_array)-1):
                if value < scale_array[i]:
                    return i - 1
        return len(scale_array)-2

    def add(self, size, data, data_function=None):
        if data_function is None:
            for d in data:
                i = self.check_index(d, self.x_scale)
                if len(self.size_scale) == 1:
                    j = 0
                else:
                    j = self.check_index(size, self.size_scale)
                self.array[j, i] += 1
        elif len(data) == 0:
            return
        else:
            try:
                value = data_function(data)
            except statistics.StatisticsError as e:
                logging.error("\nException caught: " + str(e) + "\n")
                if data_function == statistics.stdev:
                    value = 0
                else:
                    return
            i = self.check_index(value, self.x_scale)
            j = self.check_index(size, self.size_scale)
            self.array[j, i] += 1

    @staticmethod
    def get_labels(array):
        labels = [str(array[x-1]) + "; " + str(array[x]) for x in range(1, len(array))]
        labels[0] = "<" + str(labels[0]) + ")"
        labels[-1] = "<" + str(labels[-1]) + ">"
        for i in range(1, len(labels) - 1):
            labels[i] = "<" + str(labels[i]) + ")"
        return labels

    def print(self):
        print(*self.x_labels)
        for i in range(len(self.array)):
            print(self.size_labels[i], self.array[i])

    def save(self, folder, filename):
        with open(folder + filename, "w+") as file:
            file.write("," + ','.join(self.x_labels) + "\n")
            for i, row in enumerate(self.array):
                file.write(self.size_labels[i] + "," + ",".join([str(x) for x in row]) + "\n")




