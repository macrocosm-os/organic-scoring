import asyncio
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional, Sequence

import bittensor as bt

from organic_scoring.organic_queue import OrganicQueue, OrganicQueueBase
from organic_scoring.synth_dataset import SynthDatasetBase


class OrganicScoringBase(ABC):
    def __init__(
        self,
        axon: bt.axon,
        synth_dataset: SynthDatasetBase | Sequence[SynthDatasetBase],
        trigger_frequency: float | int,
        trigger: Literal["seconds", "steps"],
        trigger_frequency_min: float | int = 2,
        trigger_scaling_factor: float | int = 50,
        organic_queue: OrganicQueueBase | None = None,
    ):
        """Runs the organic weight setter task in separate threads

        Args:
            axon: The axon to use, must be started and served.
            synth_dataset: The synthetic dataset to use, must be inherited from `synth_dataset.SynthDatasetBase`.
            trigger_frequency: The frequency to trigger the organic scoring reward step.
            trigger: The trigger type, available values: "seconds", "steps".
                In case of "seconds" the `trigger_frequency` is the number of seconds to wait between each step.
                In case of "steps" the `trigger_frequency` is the number of steps to wait between each step. The
                `increment_step` method should be called to increment the step counter.
            organic_queue: The organic queue to use, must be inherited from `organic_queue.OrganicQueueBase`.
                Defaults to `organic_queue.OrganicQueue`.
            trigger_frequency_min: The minimum frequency value to trigger the organic scoring reward step.
                Defaults to 1.
            trigger_scaling_factor: The scaling factor to adjust the trigger frequency based on the size
                of the organic queue. A higher value means that the trigger frequency adjusts more slowly to changes
                in the organic queue size. This value must be greater than 0.

        Override the following methods:
            - `_on_organic_entry`: Handle an organic entry, append required values to `_organic_queue`.
                Important: this method must add the required values to the `_organic_queue`.
            - `_query_miners`: Query the miners with a given organic sample.
            - `_generate_rewards`: Concurrently generate rewards based on the sample and responses.
            - `_set_weights`: Set the weights based on generated rewards for the miners.
            - (Optional) `_generate_reference`: Generate a reference based on the sample, if required.
                Used in `_generate_rewards`.
            - (Optional) `_log_results`: Log the results.
            - (Optional) `_priority_fn`: Function with priority value for organic handles.
            - (Optional) `_blacklist_fn`: Function with blacklist for organic handles.
            - (Optional) `_verify_fn`: Function to verify requests for organic handles.

        Usage:
            1. Create a subclass of OrganicScoringBase.
            2. Implement the required methods.
            3. Create an instance of the subclass.
            4. Call the `start` method to start the organic scoring task.
            5. Call the `stop` method to stop the organic scoring task.
            6. Call the `increment_step` method to increment the step counter if the trigger is set to "steps".
        """
        self._axon = axon
        self._should_exit = False
        self._is_running = False
        self._synth_dataset = synth_dataset
        if isinstance(self._synth_dataset, SynthDatasetBase):
            self._synth_dataset = (synth_dataset,)
        self._trigger_frequency = trigger_frequency
        self._trigger = trigger
        self._trigger_min = trigger_frequency_min
        self._trigger_scaling_factor = trigger_scaling_factor
        assert self._trigger_scaling_factor > 0, "The scaling factor must be higher than 0."
        self._organic_queue = organic_queue
        if self._organic_queue is None:
            self._organic_queue = OrganicQueue()
        self._thread: Optional[threading.Thread] = None
        self._step_counter = 0
        self._step_lock = threading.Lock()

        # Optional methods to override.
        # Bittensor's internal checks require synapse to be a subclass of bt.Synapse.
        # By defining these methods as instance members, we provide flexibility for overriding them in derived classes.
        self._priority_fn: Optional[Callable[[bt.Synapse], float]] = None
        self._blacklist_fn: Optional[Callable[[bt.Synapse], tuple[bool, str]]] = None
        self._verify_fn: Optional[Callable[[bt.Synapse], bool]] = None

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
        with self._step_lock:
            if self._trigger == "steps":
                self._step_counter += 1

    def set_step(self, step: int):
        """Set the step counter to a specific value

        Args:
            step: The step value to set.
        """
        with self._step_lock:
            if self._trigger == "steps":
                self._step_counter = step

    @abstractmethod
    async def _on_organic_entry(self, synapse: bt.Synapse) -> bt.Synapse:
        """Handle an organic entry

        Important: this method must add the required values to the `_organic_queue`.

        Args:
            synapse: The synapse to handle.

        Returns:
            bt.StreamingSynapse: The handled synapse.
        """
        raise NotImplementedError

    @abstractmethod
    async def _query_miners(self, sample: Any) -> dict[str, Any]:
        """Query the miners with a sample

        Args:
            sample: The sample to query with.

        Returns:
            dict[str, Any]: The responses from the miners.
        """
        raise NotImplementedError

    @abstractmethod
    async def _generate_rewards(self, sample: Any, responses: Sequence[Any], reference: Any = None) -> dict[str, Any]:
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
    async def _set_weights(self, rewards: dict[str, Any]):
        """Set the weights for the miners

        Args:
            rewards: Dict with rewards and any additional info.
        """
        raise NotImplementedError

    async def _generate_reference(self, sample: Any) -> Optional[Any]:
        """Generate a reference based on the sample

        Args:
            sample: The sample used to generate the reference.

        Returns:
            Optional[Any]: The generated reference, if any.
        """
        return None

    async def _log_results(
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
            logs: The logs to record. Default values in the dict:
                - "time_sample": Time taken in seconds to sample the organic queue or synthetic dataset;
                - "time_responses": Time taken in seconds to concurrently query the miners and generate reference;
                - "time_rewards": Time taken in seconds to generate rewards;
                - "time_weights": Time taken in seconds to set the weights;
                - "time_total": Total time taken in seconds for the iteration;
                - "organic_queue_len": Current length of the organic queue;
                - "is_organic_sample": If the sample is from the organic queue.
            reference: The reference data.
            responses: The responses from the miners.
            rewards: The generated rewards.
            sample: The sample used.

        Returns:
            dict[str, Any]: The logs recorded.
        """
        return logs

    def _start_run_loop(self):
        """Start the run loop for the organic scoring task"""
        self._axon.attach(
            forward_fn=self._on_organic_entry,
            blacklist_fn=self._blacklist_fn,
            priority_fn=self._priority_fn,
            verify_fn=self._verify_fn,
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_loop())
        finally:
            loop.close()

    async def _run_loop(self):
        """The main loop for running the organic scoring task, either based on a time interval or steps"""
        while not self._should_exit:
            if self._trigger == "steps":
                while self._step_counter < self._trigger_frequency:
                    await asyncio.sleep(0.1)

            timer_total = time.perf_counter()

            timer_sample = time.perf_counter()
            is_organic_sample = False
            if not self._organic_queue.is_empty():
                # Choose random organic sample.
                sample = self._organic_queue.sample()
                is_organic_sample = True
            else:
                # Choose if organic queue is empty, choose random sample from provided datasets.
                sample = random.choice(self._synth_dataset).sample()

            timer_sample_elapsed = time.perf_counter() - timer_sample

            # Concurrently generate reference and query miners.
            timer_responses = time.perf_counter()
            reference_task = asyncio.create_task(self._generate_reference(sample))
            responses_task = asyncio.create_task(self._query_miners(sample))
            reference, responses = await asyncio.gather(reference_task, responses_task)
            timer_responses_elapsed = time.perf_counter() - timer_responses

            # Generate rewards.
            timer_rewards = time.perf_counter()
            rewards = await self._generate_rewards(sample, responses, reference)
            timer_rewards_elapsed = time.perf_counter() - timer_rewards

            # Set weights based on the generated rewards.
            timer_weights = time.perf_counter()
            await self._set_weights(rewards)
            timer_weights_elapsed = time.perf_counter() - timer_weights

            # Log the metrics.
            timer_elapsed = time.perf_counter() - timer_total
            logs = {
                "time_sample": timer_sample_elapsed,
                "time_responses": timer_responses_elapsed,
                "time_rewards": timer_rewards_elapsed,
                "time_weights": timer_weights_elapsed,
                "time_total": timer_elapsed,
                "organic_queue_size": self._organic_queue.size(),
                "is_organic_sample": is_organic_sample,
            }
            await self._log_results(
                logs=logs,
                reference=reference,
                responses=responses,
                rewards=rewards,
                sample=sample
            )
            await self._trigger_delay(timer_elapsed=timer_elapsed)

    async def _trigger_delay(self, timer_elapsed: float):
        """Adjust the sampling rate dynamically based on the size of the organic queue and the elapsed time.

        This method implements an annealing sampling rate that adapts to the growth of the organic queue,
        ensuring the system can keep up with the data processing demands.

        Args:
            timer_elapsed: The time elapsed during the current iteration of the processing loop. This is used 
                to calculate the remaining sleep duration when the trigger is based on seconds.

        Behavior:
            - If the trigger is set to "seconds", the method calculates a dynamic frequency based on the current queue 
            size and the scaling factor, then sleeps for the remaining duration after considering the elapsed time.
            - If the trigger is set to "steps", the method adjusts the step counter dynamically based on the current 
            queue size and the scaling factor, ensuring that the system can keep up with the processing demands.
        
        Dynamic Adjustment:
            - The `dynamic_frequency` is calculated by reducing the original frequency by a value proportional to the 
            queue size divided by the scaling factor. It ensures the frequency does not drop below `min_seconds`.
            - The `dynamic_steps` is calculated similarly, reducing the original step count by a value proportional 
            to the queue size divided by the scaling factor. It ensures the steps do not drop below `min_steps`.
        """
        # Annealing sampling rate logic
        size = self._organic_queue.size()
        if self._trigger == "seconds":
            # Adjust the sleep duration based on the queue size.
            dynamic_frequency = max(self._trigger_frequency - (size / self._trigger_scaling_factor), self._trigger_min)
            sleep_duration = max(dynamic_frequency - timer_elapsed, 0)
            await asyncio.sleep(sleep_duration)
        elif self._trigger == "steps":
            # Adjust the steps based on the queue size.
            dynamic_steps = max(self._trigger_frequency - (size // self._trigger_scaling_factor), self._trigger_min)
            with self._step_lock:
                self._step_counter = max(self._step_counter - dynamic_steps, 0)
