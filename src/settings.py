from pathlib import Path
from typing import cast, Any, Dict, List, Optional, Union
import gin
from pydantic import BaseModel, HttpUrl, root_validator
from typing import Any, Dict, List, Optional, Union
from src.models.metrics import Metric
from ray import tune


cwd = Path(__file__)
root = (cwd / "../..").resolve()


class BaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float


class Settings(BaseModel):
    datadir: Path
    testurl: HttpUrl
    trainurl: HttpUrl
    testfile: Path
    trainfile: Path
    modeldir: Path
    logdir: Path
    modelname: str
    batchsize: int


# note pydantic handles perfectly a string as url
# but mypy doesnt know that, so to keep mypy satisfied
# i am adding the "cast" for the urls.
presets = Settings(
    datadir=root / "data/raw",
    testurl=cast(
        HttpUrl,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Test_Arabic_Digit.txt",  # noqa N501
    ),
    trainurl=cast(
        HttpUrl,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt",  # noqa N501
    ),
    testfile=Path("ArabicTest.txt"),
    trainfile=Path("ArabicTrain.txt"),
    modeldir=root / "models",
    logdir=root / "logs",
    modelname="model.pt",
    batchsize=32,
)


@gin.configurable
class TrainerSettings(BaseModel):
    epochs: int
    metrics: List[Metric]
    logdir: Path
    train_steps: int
    valid_steps: int
    tunewriter: List[str] = ["tensorboard","ray"]
    optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 1e-5}
    scheduler_kwargs: Optional[Dict[str, Any]] = {"factor": 0.1, "patience": 10}
    earlystop_kwargs: Optional[Dict[str, Any]] = {
        "save": False,
        "verbose": True,
        "patience": 10,
    }


class BaseSearchSpace(BaseModel):
    input_size: int
    output_size: int
    tune_dir: Optional[Path]
    data_dir: Path

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def check_path(cls, values: Dict) -> Dict:  # noqa: N805
        datadir = values.get("data_dir")
        if not datadir.exists():
            raise FileNotFoundError(
                f"Make sure the datadir exists.\n Found {datadir} to be non-existing."
            )
        return values


# this is what ray will use to create configs
class SearchSpace(BaseSearchSpace):
    hidden_size: Union[int, SAMPLE_INT] = tune.randint(144, 256)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.05, 0.3)
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 5)
    num_heads: Union[int, SAMPLE_INT] = tune.qrandint(0, 16, 4)


# qrandint number of layers om crash te voorkomen (number heads)
