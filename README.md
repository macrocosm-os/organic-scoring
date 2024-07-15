# Organic Scoring Framework

Generic implementation for integration organic scoring into Bittensor subnet.


## Example Usage

```python
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

### Implement the following methods
- `on_organic_entry`: Handle an organic entry.
- `query_miners`: Query the miners with a given organic sample.
- `generate_rewards`: Concurrently generate rewards based on the sample and responses.
- `set_weights`: Set the weights based on generated rewards for the miners.
- (Optional) `generate_reference`: Generate a reference based on the sample.
- (Optional) `log`: Log the results.
- (Optional) `_priority_fn`: Priority value for organic handles.
- (Optional) `_blacklist_fn`: Blacklist for organic handles.


## Examples
TODO: SN1 implementation reference.
