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

    parser = argparse.ArgumentParser(description='Train_model parser')
    parser.add_argument("directory_path_in")
    args = parser.parse_args()

    directory_path_in = args.directory_path_in

    model = Model_class.MyModel()
    model.create_model()
    model.train(directory_path_in)
    model.plot_history()



if __name__ == '__main__':
    main()   