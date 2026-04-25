import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_bmine")
except importlib.metadata.PackageNotFoundError:
    pass

from . import langmodel
from . import train_util