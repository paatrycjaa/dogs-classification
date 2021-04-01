import tensorflow as tf 
from tensorflow.keras import models
from tensorflow.keras import layers


class MobileNetV3():
    def __init__(self, image_shape, lables_num):
        self.base_model = tf.keras.applications.MobileNetV3Small(input_shape=(*image_shape, 3), alpha=1.0, minimalistic=False, include_top=False,
            weights='imagenet', input_tensor=None, pooling='avg')
        self.labels_num = lables_num
        self.model = None

    def model_summary(detailed = False):
        if detailed :
            self.base_model.summary()
        self.model_summary()
    
    def build_mobileNetV3_1a(self,dropout_rate = 0.2):

        for layer in self.base_model.layers:
            layer.trainable = False
        
        self.model = models.Sequential()
        self.model.add(self.base_model)
        self.model.add(layers.Reshape((1,1,1024)))
        if dropout_rate > 0:
            self.model.add(layers.Dropout(dropout_rate))
        self.model.add(layers.Conv2D(self.labels_num, kernel_size=1, padding='same'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Activation('softmax'))

        self.model.summary()

    def train_with_generator(image_generator, epochs, optimizer = 'Adam', loss ='categorical_crossentropy', 
                                metrics =['accuracy'], callbacks = None)

        if self.model is None:
            raise AttributeError('Model is NoneType object - not build')

        self.model.compile(optimizer = optimizer, 
                        loss=loss,
                        metrics = metrics)

        self.fit(        
            image_generator.train_generator(),        
            epochs = epochs,
            callbacks = callbacks,
            validation_data = image_generator.validation_generator())

        

