import os


class PathProcessor:
    """ Class for working with paths
    """

    def __init__(self, directory=None):
        self.directory = directory
        self.file_list = self.inspect_directory(directory)

    def inspect_directory(self, directory=None):
        if directory is None:
            directory = self.directory
            if directory is None:
                raise TypeError('directory is None!')

        file_list = []
        tree = os.walk(directory)
        for d in tree:
            if d[2] is not None:
                for file in d[2]:
                    file_list.append([file, d[0]])

        return file_list

    def get_files(self):
        return self.file_list

    def sort_files(self):
        self.file_list.sort()

    def filter_by_ext(self, ext_list, file_list=None):
        if file_list is None:
            file_list = self.file_list
            if file_list is None:
                raise TypeError('file_list is None!')

        ext_tuple = tuple(ext_list)
        filtered_file_list = [file for file in file_list if file[0].endswith(ext_tuple)]

        return filtered_file_list

    @staticmethod
    def get_name_without_ext(file_name):
        name_without_ext = os.path.splitext(file_name)[0]
        return name_without_ext

    @staticmethod
    def get_ext(file_name):
        ext = os.path.splitext(file_name)[1]
        return ext
    
    @staticmethod
    def path_exist(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path {path} does not exist.')

    @staticmethod
    def get_folder_name(path_in): #enter file_path
        temp = os.path.dirname(path_in)
        temp2 = os.path.basename(temp)
        return temp2
    
    @staticmethod
    def get_paths_inside(path_in):
        path_array = []
        for root, dirs, files in os.walk(path_in): 
            for _file in files:
                path_file = os.path.join(root, _file) #path for every files in root
                path_array.append(path_file)
        return path_array
