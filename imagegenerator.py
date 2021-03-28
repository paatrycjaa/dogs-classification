import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory


class ImageGenerator():
    def __init__(self, data_path, validation_split=0.2, seed=None, batch_size=32, image_size=(256,256)):
        self.train_path = os.path.join(data_path, 'train')

        if not os.path.exists(self.train_path):
            raise FileNotFoundError('No train data in path ' + data_path)

        self.test_path = os.path.join(data_path, 'test')

        if not os.path.exists(self.test_path):
            raise FileNotFoundError('No test data in path ' + data_path)

        self.seed = seed
        self.validation_split=validation_split
        self.batch_size = batch_size
        self.image_size = image_size

    def _image_generator(self):
        return ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=self.validation_split)

    def train_generator(self):
        """Create tranining data generator.

        Returns:                
            A `DirectoryIterator` yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
        """
        return self._image_generator().flow_from_directory(
            self.train_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed,
            subset='training')

    def validation_generator(self):
        return self._image_generator().flow_from_directory(
            self.train_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed,
            subset='validation')

    def test_generator(self):
        """Create test data generator, no augmentation are applied.
        """
        return ImageDataGenerator(rescale=1./255).flow_from_directory(
            self.test_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=False)

    def train_array(self, batches_num):
        """Create numpy arrays with training data

        Args:
            batches_num (int): Number of batches to generate.

        Returns:
            (x,y) where x is np.array with images and y np.array with labels.
        """
        return generator_to_array(self.train_generator(), batches_num)

    def validation_array(self, batches_num):        
        return generator_to_array(self.validation_generator(), batches_num)

    def test_array(self, batches_num=None):
        """Create numpy arrays with test data. No augmentation is applied.
        Args:
            batches_num (int, optional): Batches limit. Defaults to None.

        Returns:
            (x,y) where x is np.array with images and y np.array with labels.
        """
        dataset = image_dataset_from_directory(
            self.test_path,
            batch_size=self.batch_size,
            label_mode='categorical',
            image_size=self.image_size,
            shuffle=False,
            validation_split=None)

        rescaled = dataset.map(_normalize_image)

        if batches_num is None:
            batches_num = len(rescaled)

        x = []
        y = []

        for x_batch, y_batch in rescaled:
            x.append(x_batch)
            y.append(y_batch)

            batches_num -= 1
            if batches_num < 1:
                break
        
        return np.concatenate(x, axis=0), np.concatenate(y, axis=0)

def generator_to_array(data_generator, batches_number):     
    x = []
    y = []
    for i in range(batches_number):
        n = next(data_generator)
        x.append(n[0])
        y.append(n[1])
    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)

def _normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255., label

if __name__ == "__main__":

    path = 'images/subset'

    generator = ImageGenerator(path)

    print(generator.train_generator())
    print(generator.validation_generator())
    print(generator.test_generator())
    
    x, y = generator.train_array(2)
    print(x.shape, y.shape)

    x, y = generator.validation_array(3)
    print(x.shape, y.shape)

    x, y = generator.test_array(4)
    print(x.shape, y.shape)


        

