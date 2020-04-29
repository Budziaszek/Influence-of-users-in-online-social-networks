import csv
import os


class FileWriter:
    file_name = "statistics_values.txt"
    folder_name = ""

    def __init__(self):
        pass

    def set_path(self, folder_name, file_name):
        self.folder_name = folder_name
        self.file_name = file_name

    def clean_file(self):
        if not os.path.exists('output/' + self.folder_name):
            os.mkdir('output/' + self.folder_name)
        open('output/' + self.folder_name + "/" + self.file_name, 'w').close()

    def write_row_to_file(self, data):
        with open('output/' + self.folder_name + "/" + self.file_name, "a+", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(data)

    def write_split_row_to_file(self, data):
        row = []
        for d in data:
            if isinstance(d, list):
                row.extend(d)
            else:
                row.append(d)
        self.write_row_to_file(row)

    def set_all(self, neighborhood_mode, file_name, labels=None):
        self.set_path(neighborhood_mode, file_name)
        self.clean_file()
        if labels is not None:
            self.write_row_to_file(labels)
