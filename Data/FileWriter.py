import csv


class FileWriter:
    file_name = "data.txt"

    def __init__(self):
        pass

    def set_file(self, file_name):
        self.file_name = file_name

    def clean_file(self):
        open('output/' + self.file_name, 'w').close()

    def write_row_to_file(self, data):
        with open('output/' + self.file_name, "a+", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(data)
