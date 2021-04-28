import analyzer
import models
import imagegenerator
import tensorflow as tf
import os

data_path = 'images/subset20'
results_path = 'results'
batch_size = 32

labels = [
    'Afghan hound',
    'basset',
    'beagle',
    'black',
    'Blenheim spaniel',
    'bloodhound',
    'bluetick',
    'borzoi',
    'Chihuahua',
    'English foxhound',
    'Irish wolfhound',
    'Japanese spaniel',
    'Maltese dog',
    'papillon',
    'Pekinese',
    'redbone',
    'Rhodesian ridgeback',
    'Shih',
    'toy terrier',
    'Walker hound'
]
img_size = (224, 224)


def train_model(model, epochs=4, model_name='model_simple', save_model=False, patience=3):
    image_generator = imagegenerator.ImageGenerator(data_path, validation_split=0.2, seed=123,
                                                    batch_size=batch_size, image_size=img_size)

    history = model.train_with_generator(image_generator, epochs, callbacks=[models.early_stopping(patience=patience)])

    an = analyzer.Analyzer(results_path)

    an.analyze_model(model, model_name, image_generator, model_parameters=None, labels=labels,
                     k=2, training_history=history, save_model=save_model)


def set_gpu_enabled(is_enabled: bool = True):
    if is_enabled:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_simple_model(**kwargs):
    model = models.MobileNetV3(img_size, len(labels))
    model.build_easy_model()
    train_model(model, **kwargs)


def run_1a(**kwargs):
    model = models.MobileNetV3(img_size, len(labels))
    model.build_mobileNetV3_1a()
    train_model(model, **kwargs)


if __name__ == "__main__":
    set_gpu_enabled(True)
    # run_simple_model(save_model=True)

    run_1a(epochs=100, model_name='1a', save_model=True, patience=20)
