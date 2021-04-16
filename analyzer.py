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

        self._save_confusion_matrix_plot(labels)

        if training_history is not None:
            self._save_training_history(training_history)

        if save_model:
            model.save(os.path.join(self.reports_path, 'model'))


    def _save_training_history(self, history):
        pd.DataFrame(history.history).to_csv(os.path.join(self.reports_path, 'history.csv'), index_label='epoch')

    def _compute_metrics(self, image_generator, labels):
        x, y_true = image_generator.test_array()
        y_predicted = self.model.predict(x)

        self.accuracy = metrics.accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_predicted, axis=1))
        self.k_accuracy = metrics.top_k_accuracy_score(np.argmax(y_true, axis=1), y_predicted, k=self.k)
        self.conf_matrix = metrics.confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_predicted, axis=1))

    def _print_metrics(self, model_name):
        print('{} metrics:'.format(model_name))
        print('accuracy: {:.3f}'.format(self.accuracy))
        print('{}-accuracy: {:.3f}'.format(self.k, self.k_accuracy))
        print('confusion matrics:\n', self.conf_matrix)

    def _generate_metrics_json(self):
        m = {
            'accuracy': self.accuracy,
            'k-accuracy_value': self.k_accuracy,
            'k-accuracy_k': self.k,
            'conf_matrix': self.conf_matrix.tolist()
        }

        with open(os.path.join(self.reports_path, 'metrics.json'), 'w') as f:
            json.dump(m, f)

    def _generate_md_report(self, model_name, timestamp, model_parameters, history):
        with open(os.path.join(self.reports_path, 'report.md'), 'w') as f:
            f.write('# {}\n*{}*\n'.format(model_name, timestamp.strftime("%Y-%m-%d %H:%M:%S")))
            self._add_model_summary(f)

            if model_parameters is not None:
                self._add_model_parameters(f, model_parameters)

            self._add_metrics(f)
            self._add_confusion_matrix(f)
            self._add_history(f, history)

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
        for k, v in items:
            text += '{} | {}\n'.format(k,v)
        f.write(text)

    def _add_model_summary(self, file):
        file.write('## Model\n```')
        self.model.summary(print_fn=lambda x : file.write(x + '\n'))
        file.write('```\n')

    def _add_history(self, file, history):
        df = pd.DataFrame(history.history)
        header = ' | '.join(df.columns)
        text = header + '\n'
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

    def _save_confusion_matrix_plot(self, labels):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('Confusion matrix')
        ax.matshow(self.conf_matrix, interpolation='nearest')
        n = np.arange(self.conf_matrix.shape[0])
        ax.set_xticks(n)
        ax.set_yticks(n)
        if labels:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        plt.savefig(os.path.join(self.reports_path, 'confusion_matrix.png'))