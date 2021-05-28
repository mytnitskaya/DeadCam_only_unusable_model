import argparse
import os
import random

class PreprocessNP():

    #def __init__(self):
        #self.path_in = path_in
        #self.proportion_to_train = proportion_to_train

    def split_in_proportion(self, path_in, proportion_to_train=0.75):
        path_train = []
        path_val = [] 
        num_class, path_class = self.get_dirs_inside(path_in)
        for _class in range(num_class):
            num_files, path_files = self.get_files_inside(path_class[_class])
            dataset_name = self.get_dataset_name(path_files)
            id_proportion_to_train = round(proportion_to_train*num_files)
            dataset = dataset_name[id_proportion_to_train-1]
            for i in range(id_proportion_to_train, len(path_files)):
                if dataset_name[i] == dataset:
                    dataset = dataset_name[i]
                else:
                    id_proportion_to_train = i
                    break
            path_train = path_train + path_files[0:id_proportion_to_train]
            path_val = path_val + path_files[id_proportion_to_train:]

        return path_train, path_val

    @staticmethod
    def get_dirs_inside(path):
        path_dirs = []
        num_dirs = 0
        for root, dirs, files in os.walk(path):
            for _dir in dirs:
                path_dir = os.path.join(root, _dir)
                path_dirs.append(path_dir)
                num_dirs = num_dirs + 1
        return num_dirs, path_dirs

    @staticmethod
    def get_files_inside(path):
        path_files = []
        num_files = 0
        for root, dirs, files in os.walk(path):
            for _file in files:
                path_file = os.path.join(root, _file) 
                path_files.append(path_file) 
                num_files = num_files + 1
        return num_files, path_files 
    
    @staticmethod
    def get_dataset_name(files):
        #dataset_name = [(os.path.basename(_file).split('_')[0] + '_' + os.path.basename(_file).split('_')[1]
        #                + '_' + os.path.basename(_file).split('_')[2]) for _file in files]
        dataset_name = []
        for _file in files:
            try:
                temp = (os.path.basename(_file).split('_')[0] + '_' + os.path.basename(_file).split('_')[1]
                            + '_' + os.path.basename(_file).split('_')[2])
                dataset_name.append(temp)
            except:
                print(_file)
        return dataset_name