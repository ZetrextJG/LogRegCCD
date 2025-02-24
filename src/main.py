import hydra

from sklearn.linear_model import LogisticRegression
from omegaconf import DictConfig
from hydra.utils import instantiate

from dataset import BaseDataset
from metrics import calculate_metrics
from utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    seed_everything(config.exp.seed)

    dataset = instantiate(config.dataset)
    train_dataset: BaseDataset = dataset(split="train")
    test_dataset: BaseDataset = dataset(split="test")
    val_dataset: BaseDataset = dataset(split="val")

    lmbda = 1.5
    lr_model = LogisticRegression(C=1 / lmbda, solver="lbfgs", max_iter=1000)
    lr_model.fit(train_dataset.get_X(), train_dataset.get_y())

    y_test = test_dataset.get_y()
    lr_y_pred = lr_model.predict(test_dataset.get_X())
    lr_metrics = calculate_metrics(y_test, lr_y_pred)
    lr_betas = lr_model.coef_
    print(lr_metrics)
    print(lr_betas)


if __name__ == "__main__":
    main()
