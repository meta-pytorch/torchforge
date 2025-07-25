# Monarch Utils

This folder contains experimental APIs built for Forge that may eventually graduate into Monarch proper.

## APIs

### Stack

The stack API allows you to combine multiple Monarch actors into a unified interface as if it was just one actor.

```
from forge.monarch_utils.stack import stack

# Stack multiple actors with a common interface
stacked = stack(actor1, actor2, actor3)

# Call methods on all actors simultaneously
stacked.method.broadcast()  # Fire-and-forget to all actors
results = stacked.method.call().get()  # Call and collect results
```

`stack` will automatically discover the the common ancestor class of all actors to determine the interface. You can also explicitly provide an interface:


```
# Auto-discover common ancestor
stacked = stack(counter1, counter2)  # Uses Counter interface

# Explicitly provide interface
stacked = stack(counter1, counter2, interface=CustomCounter)
```
