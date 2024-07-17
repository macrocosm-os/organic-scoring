import random
from typing import Any

from organic_scoring.organic_queue.organic_queue_base import OrganicQueueBase


class OrganicQueue(OrganicQueueBase):
    """Basic organic queue, implemented as a list"""
    def __init__(self):
        self._queue = []

    def add(self, sample: Any):
        """Add the sample to the queue"""
        self._queue.append(sample)

    def sample(self) -> Any:
        """Randomly pop the sample from the queue, if the queue is empty return None"""
        if self.is_empty():
            return None
        return self._queue.pop(random.randint(0, self.size() - 1))

    def size(self) -> int:
        return len(self._queue)
