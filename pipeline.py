
import analyzer
import models
import imagegenerator

data_path = 'images/subset'
results_path = 'results'
batch_size = 32

labels = ['Afghan hound', 'Maltese dog', 'Scottish deerhound']
img_size = (224,224)

def train_model(model, epochs=4, model_name='model_simple', save_model=False, patience=3):
    image_generator = imagegenerator.ImageGenerator(data_path, validation_split=0.2, seed=123,
                                                    batch_size=batch_size, image_size=img_size)

    history = model.train_with_generator(image_generator, epochs, callbacks=[models.early_stopping(patience=patience)])

    an = analyzer.Analyzer(results_path)

    an.analyze_model(model, model_name, image_generator, model_parameters=None, labels=labels,
                     k=2, training_history=history, save_model=True)


def run_simple_model(**kwargs):
    model = models.MobileNetV3(img_size, len(labels))
    model.build_easy_model()
    train_model(model, **kwargs)


def run_1a(**kwargs):
    model = models.MobileNetV3(img_size, len(labels))
    model.build_mobileNetV3_1a()
    train_model(model, **kwargs)


if __name__ == "__main__":
    run_simple_model(save_model=True)

    # run_1a(epochs=100)