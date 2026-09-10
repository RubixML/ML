# Feed Forward

Freeze the first `k` hidden layers of the network preventing their parameters from being updated during training. Useful for fine-tuning a pretrained model.

```php
public freezeFirstKLayers(int $k) : void
```

Unfreeze all hidden layers allowing their parameters to be updated during training.

```php
public unfreeze() : void
```
