from abc import ABC


class SynthDatasetBase(ABC):
    def sample(self):
        raise NotImplementedError
