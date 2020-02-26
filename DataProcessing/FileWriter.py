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

    def set_all(self, mode, file_name, labels):
        self.set_path(mode, file_name)
        self.clean_file()
        self.write_row_to_file(labels)
