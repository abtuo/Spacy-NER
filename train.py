import re
import spacy
from spacy.util import minibatch, compounding
import random
import pandas as pd
from load_data import load_data
from spacy.training.example import Example

from spacy.cli.train import train

train(config_path="checkpoint/config.cfg", output_path="output", overrides={"paths.train": "./train.spacy", "paths.dev": "./dev.spacy"})