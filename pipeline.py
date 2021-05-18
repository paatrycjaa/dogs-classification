import analyzer
import models
import imagegenerator
import tensorflow as tf
import os

data_path = 'images/subset20'
# data_path = 'images/subset'
results_path = 'results'
default_batch_size = 32

labels = [
    'Maltese_dog',
    'Shih-Tzu',
    'Afghan_hound',
    'Irish_wolfhound',
    'Saluki',
    'Scottish_deerhound',
    'Lakeland_terrier',
    'Sealyham_terrier',
    'Airedale',
    'cairn',
    'Australian_terrier',
    'Tibetan_terrier',
    'Bernese_mountain_dog',
    'EntleBucher',
    'basenji',
    'pug',
    'Leonberg',
    'Great_Pyrenees',
    'Samoyed',
    'Pomeranian'
]

# labels = [
#     'Afghan hound',
#     'Maltese dog',
#     'Scottish_deerhound'
# ]
img_size = (224, 224)


def train_model(model, epochs=4, model_name='model_simple', save_model=False, patience=3, batch_size=default_batch_size):
    image_generator = imagegenerator.ImageGenerator(data_path, validation_split=0.2, seed=123,
                                                    batch_size=batch_size, image_size=img_size)

    history = model.train_with_generator(image_generator, epochs, callbacks=[models.early_stopping(patience=patience, min_delta=1e-4)])
    # history = model.train_with_arrays(image_generator, epochs, callbacks=[models.early_stopping(patience=patience, min_delta=1e-3)])

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


def run_2a(**kwargs):
    model = models.MobileNetV3(img_size, len(labels))
    model.build_mobileNetV3_2a()
    train_model(model, **kwargs)


def run_2b(**kwargs):
    model = models.MobileNetV3(img_size, len(labels))
    model.build_mobileNetV3_2b()
    train_model(model, **kwargs)


def run_2c(**kwargs):
    model = models.MobileNetV3(img_size, len(labels))
    model.build_mobileNetV3_2c()
    train_model(model, **kwargs)


if __name__ == "__main__":
    set_gpu_enabled(True)
    # run_simple_model(save_model=True)
    run_1a(epochs=1000, model_name='1a', save_model=True, patience=10)
    run_2c(epochs=1000, model_name='2c', save_model=True, patience=20, batch_size=16)
    run_2b(epochs=1000, model_name='2b', save_model=True, patience=10)
    run_2a(epochs=1000, model_name='2a', save_model=True, patience=10)
