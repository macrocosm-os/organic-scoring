# Organic Scoring Framework

Generic implementation of organic scoring for integration into Bittensor subnet.

## Setup

Add to your requirements:
```shell
git+https://github.com/macrocosm-os/organic-scoring.git@main
```

Or install manually by:
```shell
pip install git+https://github.com/macrocosm-os/organic-scoring.git@main
```

## Implementation
### Implement the following methods
- `on_organic_entry`: Handle an organic entry.
- `query_miners`: Query the miners with a given organic sample.
- `generate_rewards`: Concurrently generate rewards based on the sample and responses.
- `set_weights`: Set the weights based on generated rewards for the miners.
- (Optional) `generate_reference`: Generate a reference based on the sample, used in `generate_rewards`.
- (Optional) `log`: Log the results.
- (Optional) `_priority_fn`: Priority value for organic handles.
- (Optional) `_blacklist_fn`: Blacklist for organic handles.


### Example Usage

```python
class OrganicScoringPrompting(OrganicScoringBase):
    # Implement all required methods.
    ...

axon = bt.axon(wallet=self.wallet, config=self.config)
axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
axon.start()

organic_scoring = OrganicScoringPrompting(
    axon=axon,
    synth_dataset=SynthDatasetLmSysChat(),
    trigger_frequency=15,
    trigger="seconds",
    validator=validator,
)
organic_scoring.start()
```


## Additional Information

TODO: SN1 implementation reference.

Feel free to reach out us through the [Bittensor discord](https://discord.gg/UqAxyhrf) (alpha 1 - SN1 channel).
