import random

import datasets

from organic_scoring.synth_dataset_base import SynthDatasetBase


class SynthDatasetLmSysChat(SynthDatasetBase):
    def __init__(self):
        self._url = "lmsys/lmsys-chat-1m"
        self.dataset = datasets.load_dataset(self._url)["train"]

    def sample(self):
        # Randomly select a sample from the dataset.
        sample_idx = random.randint(0, len(self.dataset) - 1)
        conversation = self.dataset[sample_idx]["conversation"]
        roles = [entry["role"] for entry in conversation]
        messages = [entry["content"] for entry in conversation]
        if roles[-1] == "assistant":
            roles = roles[:-1]
            messages = messages[:-1]
        return {"roles": roles, "messages": messages, "organic": False, "source": self._url}
