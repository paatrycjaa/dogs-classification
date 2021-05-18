import os
import json
import datetime
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import imagegenerator

class Analyzer:

    def __init__(self, reports_path):
        self.reports_path = reports_path

    def analyze_model(self, model, model_name, image_generator, model_parameters=None, labels=None, k=2, training_history=None, save_model=False):
        self.model = model
        self.k = k

        timestamp = datetime.datetime.now()

        self.reports_path = self._report_directory(model_name, timestamp)

        self._compute_metrics(image_generator, labels)
        self._print_metrics(model_name)

        self._generate_md_report(model_name, timestamp, model_parameters, training_history)
        self._generate_metrics_json()
        self._dump_sklearn_report()

        if training_history is not None:
            self._save_training_history(training_history)

        if save_model:
            model.save(os.path.join(self.reports_path, 'model'))


    def _save_training_history(self, history):
        pd.DataFrame(history.history).to_csv(os.path.join(self.reports_path, 'history.csv'), index_label='epoch')

    def _compute_metrics(self, image_generator, labels):
        x, y_true = image_generator.test_array()
        y_predicted = self.model.predict(x)

        y_true1d = np.argmax(y_true, axis=1)
        y_predicted1d = np.argmax(y_predicted, axis=1)

        self.accuracy = metrics.accuracy_score(y_true1d, y_predicted1d)
        self.k_accuracy = metrics.top_k_accuracy_score(y_true1d, y_predicted, k=self.k)
        self.conf_matrix = metrics.confusion_matrix(y_true1d, y_predicted1d)

        self.sklearn_metrics = metrics.classification_report(y_true1d, y_predicted1d, target_names=labels, output_dict=True, digits=3)
        self.sklearn_metrics_printable = metrics.classification_report(y_true1d, y_predicted1d, target_names=labels, output_dict=False, digits=3)

    def _print_metrics(self, model_name):
        print('{} metrics:'.format(model_name))
        print('accuracy: {:.3f}'.format(self.accuracy))
        print('{}-accuracy: {:.3f}'.format(self.k, self.k_accuracy))
        print('confusion matrics:\n', self.conf_matrix)
        print('sklearn report:\n', self.sklearn_metrics_printable)

    def _generate_metrics_json(self):
        m = {
            'accuracy': self.accuracy,
            'k-accuracy_value': self.k_accuracy,
            'k-accuracy_k': self.k,
            'conf_matrix': self.conf_matrix.tolist()
        }

        with open(os.path.join(self.reports_path, 'metrics.json'), 'w') as f:
            json.dump(m, f)

    def _dump_sklearn_report(self):
        with open(os.path.join(self.reports_path, 'sklearn_metrics.json'), 'w') as f:
            json.dump(self.sklearn_metrics, f)

    def _generate_md_report(self, model_name, timestamp, model_parameters, history):
        with open(os.path.join(self.reports_path, 'report.md'), 'w') as f:
            f.write('# {}\n*{}*\n'.format(model_name, timestamp.strftime("%Y-%m-%d %H:%M:%S")))
            self._add_model_summary(f)

            if model_parameters is not None:
                self._add_model_parameters(f, model_parameters)

            self._add_metrics(f)
            self._add_sklearn_report(f)
            self._add_confusion_matrix(f)
            self._add_history(f, history)

    def _add_sklearn_report(self, file):
        text = '## Sklearn report\n```\n{}\n```\n'.format(self.sklearn_metrics_printable)
        file.write(text)

    def _add_confusion_matrix(self, file):
        text = '## Confusion matrix\n'

        class_num = self.conf_matrix.shape[0]
        text += ' | '.join([str(x) for x in range(1, class_num+1)]) + '\n'
        text += ' | '.join(['---']*class_num) + '\n'
        for row in self.conf_matrix:
            text += ' | '.join([str(x) for x in row])
            text += '\n'
        file.write(text)

    def _add_metrics(self, file):
        text = '## Metrics\n'
        text += '| Metric | Value \n --- | ---\n'
        text += ' {} | {:.3f} \n'.format('accuracy', self.accuracy)
        text += ' {}-{} | {:.3f} \n'.format(self.k, 'accuracy', self.k_accuracy)
        file.write(text)


    def _add_model_parameters(self, file, parameters):
        text = '### Model parameters\n'
        text += '| Prameters | Value \n --- | ---\n'
        for k, v in parameters.items():
            text += '{} | {}\n'.format(k,v)
        f.write(text)

    def _add_model_summary(self, file):
        file.write('## Model\n```')
        self.model.summary(print_fn=lambda x : file.write(x + '\n'))
        file.write('```\n')

    def _add_history(self, file, history):
        df = pd.DataFrame(history.history)
        text = '## Training History\n'
        header = ' | '.join(df.columns)
        text += header + '\n'
        text += ' | '.join(len(df.columns)*['---']) + '\n'
        for _, row in df.iterrows():
            text += ' | '.join([str(round(x,4)) for x in row.values])
            text += '\n'
        file.write(text)

    def _report_directory(self, model_name, timestamp):
        dir_name = 'results_{}_{}_{}/'.format(model_name, timestamp.strftime("%m-%d_%H-%M"), uuid.uuid1().hex[:7])
        path = os.path.join(self.reports_path, dir_name)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        return path