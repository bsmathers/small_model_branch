from metamon.rl.pretrained import (
    LocalPretrainedModel,
    PretrainedModel,
    LocalFinetunedModel,
)
from metamon.rl.evaluate import (
    pretrained_vs_pokeagent_ladder,
    pretrained_vs_local_ladder,
    pretrained_vs_baselines,
)

import os

MODEL_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "models")
TRAINING_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "training")
DATASET_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "datasets")
