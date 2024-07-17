from abc import ABC
from typing import Any


class SynthDatasetBase(ABC):
    def sample(self) -> Any:
        raise NotImplementedError
