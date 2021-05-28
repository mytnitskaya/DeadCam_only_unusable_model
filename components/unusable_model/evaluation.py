import tensorflow as tf

def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    import os, sys
    import argparse
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    from components.unusable_model import my_model as Model_class

    parser = argparse.ArgumentParser(description='Path for evaluation data')
    parser.add_argument('directory_path_in')
    parser.add_argument('-p', dest='path_to_model_file', default='save/best_model/unusable_model.hdf5')
    args = parser.parse_args()

    directory_path_in = args.directory_path_in
    path_to_model_file = args.path_to_model_file

    model = Model_class.MyModel()
    model.load(path_to_model_file)
    model.evaluate(directory_path_in)
 



if __name__ == '__main__':
    main() 