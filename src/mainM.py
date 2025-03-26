import hydra
from scipy.stats import alpha

from sklearn.linear_model import LogisticRegression
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np
from plots import plot_metrics, plot_betas
from dataset import BaseDataset
from metrics import calculate_metrics
from utils import seed_everything
from logregCCD import LogRegCCD
from matplotlib import pyplot as plt
import os
from metrics import MetricsCollated


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    seed_everything(config.exp.seed)

    dataset = instantiate(config.dataset)
    train_dataset: BaseDataset = dataset(split="train")
    test_dataset: BaseDataset = dataset(split="test")
    val_dataset: BaseDataset = dataset(split="val")

    # alpha = [0.001, 0.01, 0.05, 0.1,0.3,0.5,0.7, 1.0]
    alpha = [1]
    for a in alpha:
        ccd_model = LogRegCCD(
            alpha=a,  # lasso
            heuristic_intercept=False,
            fit_intercept=False,
        )
        # tutaj  _fit
        results = ccd_model.fit(
            train_dataset.get_X(),
            train_dataset.get_y(),
            val_dataset.get_X(),
            val_dataset.get_y(),
        )
        print(results)
        # scores = [result["metrics"] for result in results]
        # betas = [result["betas"] for result in results]
        # betas = np.stack(betas)
        # print(scores)
        # print([score['roc_auc'] for score in scores])
        # print(betas)
        print(ccd_model.lmbda)
        print(ccd_model.beta0)
        print(ccd_model.betas)

    filename = f"CCD_alpha{a}.txt"
        # with open(filename, "w") as f:
        #     pass
        # with open(filename, "w") as f:
        #     print(str(results))
        #     f.write(str(results))

    # lmbda = ccd_model.lmbda

    lr_model = LogisticRegression(
        penalty=None
    )
    lr_model.fit(train_dataset.get_X(), train_dataset.get_y())
    y_pred = lr_model.predict(test_dataset.get_X())
    print(calculate_metrics(test_dataset.get_y(), y_pred))
    print(lr_model.intercept_)
    print(lr_model.coef_)

    # num = [k for k in range(betas.shape[0])]
    # metr = plot_betas(betas, num)
    # plt.show()

if __name__ == "__main__":
    main()
