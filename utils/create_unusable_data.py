import random
import os, sys
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import argparse
from tqdm import tqdm
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from components.common import path_lib

EXTENSIONS_VIDEO = ['.mov', '.mp4']
EXTENSIONS_IMAGE = ['.jpg', '.png']

def get_output_path_video(file_path_in, path_out, class_name):
    file_folder = path_lib.PathProcessor.get_folder_name(file_path_in)
    folder_path_out = os.path.join(path_out, (file_folder + '_' + class_name))
    os.makedirs(folder_path_out, exist_ok = True)
    file_path_out = os.path.join(folder_path_out, (os.path.basename(os.path.splitext(file_path_in)[0])+'.mp4'))
    return file_path_out

def get_output_path_image(file_path_in, path_out, class_name):
    file_folder = path_lib.PathProcessor.get_folder_name(file_path_in)
    folder_path_out = os.path.join(path_out, (file_folder + '-' + class_name))
    os.makedirs(folder_path_out, exist_ok = True)
    file_path_out = os.path.join(folder_path_out, os.path.basename(file_path_in))
    return file_path_out

def apply_filter(image, parameters):
    seq = iaa.Sequential()
    if 'blur' in parameters['filters']:
        seq.add(iaa.GaussianBlur(sigma=parameters['blur']))
    if 'jpeg_compression' in parameters['filters']:
        seq.add(iaa.AveragePooling(parameters['pooling']))
        seq.add(iaa.JpegCompression(compression=parameters['jpeg']))
    if 'dark' in parameters['filters']:
        seq.add(iaa.Add(parameters['dark']))

    images_aug = seq(image=image)
    return images_aug

def generat_filter_parameters():
    return {'filters': set(random.choices(['blur', 'jpeg_compression', 'dark'], k=random.randint(1, 3))),
        'blur': random.randint(16, 24),
        'pooling': random.randint(8, 15),
        'jpeg': random.randint(90, 95),
        'dark': random.randint(-220, -170)}

def process_video_filter(file_path_in, file_path_out):
    video_in = cv2.VideoCapture(file_path_in)
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video_in.get(cv2.CAP_PROP_FPS))
    video_out = cv2.VideoWriter(file_path_out, cv2.VideoWriter_fourcc(*'mp4v'), length, (width, height))
    parameters = generat_filter_parameters()
    print(parameters)
    while video_in.isOpened():
        success,image = video_in.read()
        if not success:
            break
        image_filter = apply_filter(image, parameters)
        video_out.write(image_filter)
    video_in.release()
    video_out.release()

def process_image_filter(file_path_in, file_path_out):
    image_in = cv2.imread(file_path_in)
    parameters = generat_filter_parameters()
    image_filter = apply_filter(image_in, parameters)
    cv2.imwrite(file_path_out, image_filter)

def main():
    parser = argparse.ArgumentParser(description = 'Create_unusable_image parser')
    parser.add_argument('num_files_to_enter', type = int, help = 'number of files to enter for creating unusable images')
    parser.add_argument('path_in', help = 'path to original image folder')
    parser.add_argument('path_out', help = 'path to return unusable image')
    args = parser.parse_args()

    path_in = args.path_in
    path_lib.PathProcessor.path_exist(path_in)
    path_out = args.path_out
    path_lib.PathProcessor.path_exist(path_out)
    n_files = args.num_files_to_enter

    files = path_lib.PathProcessor.get_paths_inside(path_in)
    if n_files > len(files):
        print('you entered', n_files, 'for num_files_to_enter.', 
        'it bigger than count of files in directory. we use max count of file for this directory =', len(files))
        n_files = len(files)

    #files_random = random.choices(files, k=n_files)
    files_random = random.sample(files, k=n_files)
    class_name = os.path.basename(path_in)
    for file_path_in in tqdm(files_random):
        file_format = path_lib.PathProcessor.get_ext(file_path_in)
        if file_format in EXTENSIONS_VIDEO:
            file_path_out = get_output_path_video(file_path_in, path_out, class_name)
            process_video_filter(file_path_in, file_path_out)
        if file_format in EXTENSIONS_IMAGE:
            file_path_out = get_output_path_image(file_path_in, path_out, class_name)
            process_image_filter(file_path_in, file_path_out)
        
if __name__ == '__main__':
    main()