import hydra
from omegaconf import DictConfig

from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../configs", config_name="test_dataset")
def main(config: DictConfig):
    dataset = instantiate(config.dataset)
    train_dataset = dataset(split="train")
    print(train_dataset.get_X())


if __name__ == "__main__":
    main()
