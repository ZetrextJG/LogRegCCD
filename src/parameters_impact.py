"""
Exploration of impact of parameters of generation of synthetic dataset on results of CCD Logistic Regression model.
"""
import hydra
from omegaconf import DictConfig
import numpy as np

from dataset import SyntheticDataset
from metrics import calculate_metrics
from utils import seed_everything
from logregCCD import LogRegCCD
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger(__name__)


def experiment(parameter_name):
    """
    Experiments for certain parameter:
    p - probability of positive class
    n - number of observations
    d - number of features
    g - correlation between features
    """
    # Default values
    p = 0.5
    n = 2000
    d = 10
    g = 0.1

    if parameter_name == 'p':
        scope = np.linspace(0.2, 0.8, 60)
    elif parameter_name == 'n':
        scope = np.linspace(100, 5000, dtype=int)
    elif parameter_name == 'd':
        scope = np.linspace(2, 200, num=20, dtype=int)
    elif parameter_name == 'g':
        scope = np.linspace(0.1, 0.9, num=90)
    else:
        raise ValueError('Wrong parameter name')
    balanced_accuracy = []
    roc_auc = []
    for parameter in scope:
        if parameter_name == 'p':
            train_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=parameter, n=n, d=d, g=g)
            validation_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=parameter, n=n, d=d, g=g,
                                                  split='val')
            test_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=parameter, n=n, d=d, g=g, split='test')
        elif parameter_name == 'n':
            train_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=p, n=parameter, d=d, g=g)
            validation_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=p, n=parameter, d=d, g=g,
                                                  split='val')
            test_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=p, n=parameter, d=d, g=g, split='test')
        elif parameter_name == 'd':
            train_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=p, n=n, d=parameter, g=g)
            validation_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=p, n=n, d=parameter, g=g,
                                                  split='val')
            test_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=p, n=n, d=parameter, g=g, split='test')
        elif parameter_name == 'g':
            train_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=p, n=n, d=d, g=parameter)
            validation_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=p, n=n, d=d, g=parameter,
                                                  split='val')
            test_dataset = SyntheticDataset(name='synthetic', num_classes=2, p=p, n=n, d=d, g=parameter, split='test')
        X_train, y_train = train_dataset.get_X(), train_dataset.get_y()
        X_test, y_test = test_dataset.get_X(), test_dataset.get_y()
        X_val, y_val = validation_dataset.get_X(), validation_dataset.get_y()
        model = LogRegCCD(alpha=1)
        model.fit(X_train, y_train, X_val, y_val)
        test_metrics = calculate_metrics(
            y_test, model.predict_proba(X_test), prefix="test"
        )
        balanced_accuracy.append(test_metrics['test_balanced_acc'])
        roc_auc.append(test_metrics['test_roc_auc'])
    plt.plot(scope, balanced_accuracy, label='Balanced accuracy')
    plt.plot(scope, roc_auc, label='ROC AUC')
    plt.ylabel('Metric')
    if parameter_name == 'p':
        plt.xlabel('Probability of positive class')
        plt.title('Metrics vs Probability of positive class')
    elif parameter_name == 'n':
        plt.xlabel('Number of observations')
        plt.title('Metrics vs Number of observations')
    elif parameter_name == 'd':
        plt.xlabel('Number of features')
        plt.title('Metrics vs Number of features')
    elif parameter_name == 'g':
        plt.xlabel('Correlation between features')
        plt.title('Metrics vs Correlation between features')
    plt.legend()
    plt.savefig(f'{parameter_name}.png')
    plt.clf()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    Experiments
    """
    # Allow reproducibility
    logger.info("Setting seed")
    seed_everything(config.exp.seed)

    # Probability
    experiment('p')

    # Number of observations
    experiment('n')

    # Number of features
    experiment('d')

    # Correlation
    experiment('g')


if __name__ == "__main__":
    main()