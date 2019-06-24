### Wrapper
Wrappers are meta-estimators that wrap a base estimator for the purposes of adding extra functionality. Most wrappers allow access to the underlying base estimator's methods from the Wrapper instance, but you can also return the base estimator directly using the `base()` method.

To return the base estimator:
```php
public base() : Estimator
```

**Example:**

```php
$base = $estimator->base();
```