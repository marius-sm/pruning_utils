## This repo implements a prunable Conv2D layer

### PrunableConv

This layer extends tf.keras.layers.Conv2D. It provide several handy functions for pruning.
"Spatial" pruning : 
Let k be the kernel of the convolutions, of shape (h, w, n_in, n_out)
Let m be the pruning mask, of shape (n_in, n_out)
- prune_n_connexions(n, criteria) : this will prune values of m
- prune_n_outputs(n, criteria) : this will prune m equally for each row, such that each channel is always connected to the same number of filters.
"Depthwise" pruning :
- prune_n_depthwise(n, criteria) : this will prune pixels of the kernel.

Criteria can be any of "magnitude" (lowest magnitude weights are pruned ; for filters we take the L1 norm) or "random"

Additionnaly, it provides a way to specify a pruning schedule.

Example :
```python
from custom_layers import PrunableConv

settings = {
    "pruning_type": "connexions", # or "outputs"
    "num_steps": 8, # number of total pruning steps
    "criteria": "magnitude", # or "random"
    "schedule_type": "geometric" # or "linear",
    "initial_value": 32*32,
    "final_value": 32
}

x = ...
x = PrunableConv(filters=32, kernel_size=3, padding="same", activation="relu",
    auto_prune_settings=settings
)(x)
```

It the case where output-pruning is used and n_in == n_out, one can also specify "min_matching" as the pruning criteria. This will iteratively prune one-to-one matchings of minimum norm.