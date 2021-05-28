import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os, sys, random, datetime
import seaborn as sns
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from components.common import path_lib
from components.common import data_preprocessor_lib
from components.unusable_model import data_generator as DataGenerator
from components.unusable_model import preprocess_np_data

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.model = tf.keras.Model()
        self.base_model = tf.keras.Model()
        self.optimizer_RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        self.optimizer_Adam = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.data_factory = data_preprocessor_lib.DataPreprocessorFactory('usable_model')
        self.preprocessor = self.data_factory.create_data_preprocessor()  
        self.preprocessor_np = preprocess_np_data.PreprocessNP()
        self.save = self.get_path_to_save()

    def create_model(self):
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
        self.base_model.trainable = False
        inputs = self.base_model.input
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer_Adam, 
                            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), 
                            tf.keras.metrics.Recall()])

    def train(self,data_directory):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        train, val = self.preprocessor_np.split_in_proportion(data_directory)
        train = random.sample(train, len(train))
        val = random.sample(val, len(val))
        train = DataGenerator.DataGenerator(train)
        val =  DataGenerator.DataGenerator(val)
        os.makedirs(self.save)
        checkpoint_path = os.path.join(self.save, 'model.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')
        callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                            save_best_only=True, verbose=1)
        self.history = self.model.fit(x = train, validation_data=val,
                            epochs=40, batch_size=128, 
                            class_weight = {0: 3, 1: 1}, callbacks = [callback]) 
    
    def plot_history(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.history.history['loss'])
        axs[0, 0].plot(self.history.history['val_loss'])
        axs[0, 0].set_title('model loss')
        axs[0, 0].legend(['train', 'val'], loc='upper left')
        axs[0, 1].plot(self.history.history['binary_accuracy'])
        axs[0, 1].plot(self.history.history['val_binary_accuracy'])
        axs[0, 1].set_title('model accuracy')
        axs[0, 1].legend(['train', 'val'], loc='lower right')
        axs[1, 0].plot(self.history.history['precision'])
        axs[1, 0].plot(self.history.history['val_precision'])
        axs[1, 0].set_title('model precision')
        axs[1, 0].set(xlabel = 'epoch')
        axs[1, 0].legend(['train', 'val'], loc='lower right')
        axs[1, 1].plot(self.history.history['recall'])
        axs[1, 1].plot(self.history.history['val_recall'])
        axs[1, 1].set_title('model recall')
        axs[1, 1].set(xlabel = 'epoch')
        axs[1, 1].legend(['train', 'val'], loc='lower right')
        for ax in axs.flat:
            ax.label_outer()
        path_to_save = os.path.join(self.save, 'model_history.png')
        plt.savefig(path_to_save)
        plt.show()

    def load(self, path_to_model_file):
        self.model = load_model(path_to_model_file)
    
    def evaluate(self, data_directory):
        data = path_lib.PathProcessor.get_paths_inside(data_directory)
        data = random.sample(data, len(data))
        test = DataGenerator.DataGenerator(data)
        #result = self.model.evaluate(test, verbose = 1)
        self.metrics(test)

    def inference(self, data_np):  
        data = self.preprocessor.resize_video(data_np)
        data = np.array(self.preprocessor.preprocessed_video(data))
        preds = self.model.predict(data, verbose = 0) 
        preds = self.predict(preds)
        return preds

    def metrics(self, data):
        y_preds = self.model.predict(data, verbose = 0)
        y_preds = [pred[0] for pred in y_preds]
        y_preds = self.round_array(y_preds)
        y_true = []
        for _data in data:
            for label in _data[1]:
                y_true.append(label)

        accuracy = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds)
        recall = recall_score(y_true, y_preds)
        f1 = f1_score(y_true, y_preds)
        print('accuracy:', accuracy, '\nprecision:', precision, '\nrecall:', recall, '\nf1:', f1 ) 
        self.plot_confusion_matrix(y_true, y_preds)

    @staticmethod
    def plot_confusion_matrix(y_true, y_preds):
        matrix = confusion_matrix(y_true, y_preds)
        categories = ['Непригодно','Пригодно']
        sns.heatmap(matrix, annot = True, cmap = 'Reds', xticklabels = categories, 
                    yticklabels = categories)
        plt.xlabel('Ответ модели')
        plt.ylabel('Класс видео')
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'confusion_matrix.png'))
        plt.show()

    @staticmethod
    def predict(predictions, collected_factor=0.7, new_factor=0.3):
        #decreasing
        result = predictions[0]
        for i in range(1, len(predictions)):
            result = result * collected_factor + predictions[i] * new_factor
        return result

    @staticmethod
    def round_array(data, factor = 0.5):
        data_labels = []
        for var in data:
            if var >= factor:
                data_labels.append(1)
            else:
                data_labels.append(0)
        return data_labels
    
    @staticmethod
    def get_path_to_save():
        path_folder = os.path.dirname(os.path.abspath(__file__))
        date = datetime.datetime.now()
        date = date.strftime("%Y-%m-%d.%H-%M")

        path = os.path.join(path_folder, 'save', 'save.%s') % date
        #os.makedirs(path)
        return path