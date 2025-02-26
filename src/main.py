import hydra

from sklearn.linear_model import LogisticRegression
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np

from dataset import BaseDataset
from metrics import calculate_metrics
from utils import seed_everything
from logregCCD import LogRegCCD


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    seed_everything(config.exp.seed)

    dataset = instantiate(config.dataset)
    train_dataset: BaseDataset = dataset(split="train")
    test_dataset: BaseDataset = dataset(split="test")
    val_dataset: BaseDataset = dataset(split="val")

    ccd_model = LogRegCCD(
        alpha=1,  # lasso
        heuristic_intercept=False,
        fit_intercept=False,
    )
    results = ccd_model.fit(
        train_dataset.get_X(),
        train_dataset.get_y(),
        val_dataset.get_X(),
        val_dataset.get_y(),
    )
    scores = [result["score"] for result in results]
    betas = [result["betas"] for result in results]
    betas = np.stack(betas)
    print(scores)
    print(betas)
    print(ccd_model.lmbda)
    print(ccd_model.beta0)
    print(ccd_model.betas)

    lmbda = ccd_model.lmbda
    lr_model = LogisticRegression(
        C=1 / lmbda,
        penalty="l1",
        solver="liblinear",
        max_iter=1000,
        fit_intercept=False,
    )
    lr_model.fit(train_dataset.get_X(), train_dataset.get_y())

    print(lr_model.intercept_)
    print(lr_model.coef_)


if __name__ == "__main__":
    main()
