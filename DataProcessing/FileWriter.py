import csv
import os


class FileWriter:
    OUTPUT = 'output'
    CLUSTERING = 'clustering'
    CORRELATION = 'correlation'
    TABLE = 'table'
    STATISTICS = 'data_statistics'
    STABILITY = 'stability'

    def __init__(self):
        if not os.path.exists(FileWriter.OUTPUT):
            os.mkdir(FileWriter.OUTPUT)

    @staticmethod
    def make_dir(sub_folder):
        path = os.path.join(FileWriter.OUTPUT, sub_folder)
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def clean_file(sub_folder, file_name):
        path = os.path.join(FileWriter.OUTPUT, sub_folder, file_name)
        FileWriter.make_dir(sub_folder)
        open(path, 'w').close()

    @staticmethod
    def write_row_to_file(sub_folder, file_name, data):
        path = os.path.join(FileWriter.OUTPUT, sub_folder, file_name)
        FileWriter.make_dir(sub_folder)
        with open(path, "a+", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(data)

    @staticmethod
    def write_split_row_to_file(sub_folder, file_name, data):
        row = []
        for d in data:
            if isinstance(d, list):
                row.extend(d)
            else:
                row.append(d)
        FileWriter.write_row_to_file(sub_folder, file_name, row)

    @staticmethod
    def write_dict_to_file(sub_folder, file_name, dict):
        for key in dict:
            row = [str(key), str(dict[key])]
            FileWriter.write_row_to_file(sub_folder, file_name, row)

    @staticmethod
    def write_data_frame_to_file(sub_folder, file_name, df):
        FileWriter.make_dir(sub_folder)
        path = os.path.join(FileWriter.OUTPUT, sub_folder, file_name)
        df.to_csv(path)

    @staticmethod
    def get_path(sub_folder, file_name):
        FileWriter.make_dir(sub_folder)
        return os.path.join(FileWriter.OUTPUT, sub_folder, file_name)
