import asyncio
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Sequence

import bittensor as bt
from prompting.organic.organic_scoring.synth_dataset_base import SynthDatasetBase


class OrganicScoringBase(ABC):
    def __init__(
        self,
        axon: bt.axon,
        synth_dataset: SynthDatasetBase | Sequence[SynthDatasetBase],
        trigger_frequency: float | int,
        trigger: Literal["seconds", "steps"],
        *args, **kwargs,
    ):
        """Runs the organic weight setter task in separate threads

        Args:
            axon: The axon to use, must be started and served.
            synth_dataset: The synthetic dataset to use.
            trigger_frequency: The frequency to trigger the organic scoring reward step.
            trigger: The trigger type, available values: "seconds", "steps".
                In case of "seconds" the `trigger_frequency` is the number of seconds to wait between each step.
                In case of "steps" the `trigger_frequency` is the number of steps to wait between each step. The
                `increment_step` method should be called to increment the step counter.

        Override the following methods:
            _priority_fn: Priority value for organic handles.
            _blacklist_fn: Blacklist for organic handles.
            on_organic_entry: Handle an organic entry.
            query_miners: Query the miners with a given organic sample.
            generate_rewards: Concurrently generate rewards based on the sample and responses.
            set_weights: Set the weights based on generated rewards for the miners.
            (Optional) generate_reference: Generate a reference based on the sample.
            (Optional) log: Log the results of the scoring task.
        """
        self._axon = axon
        self._should_exit = False
        self._is_running = False
        self._synth_dataset = synth_dataset
        if not isinstance(self._synth_dataset, (list, tuple)):
            self._synth_dataset = tuple(synth_dataset)
        self._trigger_frequency = trigger_frequency
        self._trigger = trigger
        self._thread: Optional[threading.Thread] = None
        self._organic_queue = []
        self._step_counter = 0

    def start(self):
        """Start the organic scoring in a background thread"""
        if not self._is_running:
            bt.logging.debug("Starting organic tasks in background thread.")
            self._should_exit = False
            self._is_running = True
            self._thread = threading.Thread(target=self._start_run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the organic scoring background thread"""
        if self._is_running:
            bt.logging.debug("Stopping organic tasks in background thread.")
            self._should_exit = True
            self._is_running = False
            self._thread.join()

    def increment_step(self):
        """Increment the step counter if the trigger is set to `steps`"""
        if self._trigger == "steps":
            self._step_counter += 1

    def set_step(self, step: int):
        """Set the step counter to a specific value

        Args:
            step: The step value to set.
        """
        if self._trigger == "steps":
            self._step_counter = step

    @abstractmethod
    async def on_organic_entry(self, synapse: bt.StreamingSynapse) -> bt.StreamingSynapse:
        """Handle an organic entry

        Args:
            synapse: The synapse to handle.

        Returns:
            bt.StreamingSynapse: The handled synapse.
        """
        raise NotImplementedError

    async def _priority_fn(self, synapse: bt.StreamingSynapse) -> float:
        """Priority function for the axon"""
        return 1000000.0

    async def _blacklist_fn(self, synapse: bt.StreamingSynapse) -> tuple[bool, str]:
        """Blacklist function for the axon"""
        return False, ""

    def _start_run_loop(self):
        """Start the run loop for the organic scoring task"""
        self._axon.attach(
            forward_fn=self.on_organic_entry,
            blacklist_fn=self._blacklist_fn,
            priority_fn=self._priority_fn,
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_loop())
        finally:
            loop.close()

    @abstractmethod
    async def query_miners(self, sample: Any) -> dict[str, Any]:
        """Query the miners with a sample

        Args:
            sample: The sample to query with.

        Returns:
            dict[str, Any]: The responses from the miners.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_rewards(self, sample: Any, responses: Sequence[Any], reference: Any = None) -> dict[str, Any]:
        """Generate rewards based on the sample and responses

        Args:
            sample: The sample to use.
            responses: The responses from the miners.
            reference: The reference data. Defaults to None.

        Returns:
            dict[str, Any]: The generated rewards information.
        """
        raise NotImplementedError

    @abstractmethod
    async def set_weights(self, rewards: dict[str, Any]):
        """Set the weights for the miners

        Args:
            rewards: Dict with rewards and any additional info.
        """
        raise NotImplementedError

    async def generate_reference(self, sample: Any) -> Optional[Any]:
        """Generate a reference based on the sample

        Args:
            sample: The sample used to generate the reference.

        Returns:
            Optional[Any]: The generated reference, if any.
        """
        return None

    async def log(
        self,
        logs: dict[str, Any],
        reference: Any,
        responses: dict[str, Any],
        rewards: dict[str, Any],
        sample: Any,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Log the results of the organic scoring iteration

        Args:
            logs: The logs to record.
            reference: The reference data.
            responses: The responses from the miners.
            rewards: The generated rewards.
            sample: The sample used.

        Returns:
            dict[str, Any]: The logs recorded.
        """
        return logs

    async def _run_loop(self):
        """The main loop for running the organic scoring task, either based on a time interval or steps"""
        while not self._should_exit:
            if self._trigger == "steps":
                while self._step_counter < self._trigger_frequency:
                    await asyncio.sleep(0.1)

            timer_total = time.perf_counter()

            timer_sample = time.perf_counter()
            if self._organic_queue:
                # Choose random organic sample.
                sample = self._organic_queue.pop(random.randint(0, len(self._organic_queue) - 1))
            else:
                # Choose if organic queue is empty, choose random sample from provided datasets.
                sample = random.choice(self._synth_dataset).sample()

            timer_sample_elapsed = time.perf_counter() - timer_sample

            # Concurrently generate reference and query miners.
            timer_responses = time.perf_counter()
            reference_task = asyncio.create_task(self.generate_reference(sample))
            responses_task = asyncio.create_task(self.query_miners(sample))
            reference, responses = await asyncio.gather(reference_task, responses_task)
            timer_responses_elapsed = time.perf_counter() - timer_responses

            # Generate rewards.
            timer_rewards = time.perf_counter()
            rewards = await self.generate_rewards(sample, responses, reference)
            timer_rewards_elapsed = time.perf_counter() - timer_rewards

            # Set weights based on the generated rewards.
            timer_weights = time.perf_counter()
            await self.set_weights(rewards)
            timer_weights_elapsed = time.perf_counter() - timer_weights

            # Log the metrics.
            timer_elapsed = time.perf_counter() - timer_total
            logs = {
                "time_sample": timer_sample_elapsed,
                "time_responses": timer_responses_elapsed,
                "time_rewards": timer_rewards_elapsed,
                "time_weights": timer_weights_elapsed,
                "time_total": timer_elapsed,
                "organic_queue_len": len(self._organic_queue),
            }
            await self.log(
                logs=logs,
                reference=reference,
                responses=responses,
                rewards=rewards,
                sample=sample
            )

            if self._trigger == "seconds":
                await asyncio.sleep(self._trigger_frequency)
            elif self._trigger == "steps":
                self._step_counter = 0
