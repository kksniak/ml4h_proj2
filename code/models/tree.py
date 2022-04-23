import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import load_config
from evaluation import evaluate

config = load_config()

model_name = 'Bio_ClinicalBERT'
dataset_id = 'small_balanced'


class Tree:

    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100,
                                                 verbose=3,
                                                 n_jobs=-1)

    def load_data(self):
        X_train_path = os.path.join(
            config['DATA_PATH'], f'X_feat_{model_name}_{dataset_id}_train.npy')
        y_train_path = os.path.join(config['DATA_PATH'],
                                    f'y_{model_name}_{dataset_id}_train.npy')
        X_test_path = os.path.join(
            config['DATA_PATH'], f'X_feat_{model_name}_{dataset_id}_test.npy')
        y_test_path = os.path.join(config['DATA_PATH'],
                                   f'y_{model_name}_{dataset_id}_test.npy')

        with open(X_train_path, 'rb') as f:
            self.X_train = np.load(f)

        with open(y_train_path, 'rb') as f:
            self.y_train = np.load(f)

        with open(X_test_path, 'rb') as f:
            self.X_test = np.load(f)

        with open(y_test_path, 'rb') as f:
            self.y_test = np.load(f)

    def train(self):
        print('Training...')
        self.classifier.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.classifier.predict_proba(self.X_test)
        evaluate(f'{model_name}_{str(self.classifier)}',
                 y_pred,
                 self.y_test,
                 save_results=True)


if __name__ == '__main__':
    tree = Tree()
    tree.load_data()
    tree.train()
    tree.evaluate()