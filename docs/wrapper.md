# Wrapper
Wrappers are meta-estimators that wrap a base estimator for the purposes of adding extra functionality. All wrappers allow access to the underlying base estimator's methods from the Wrapper instance, but you can also return and use the base estimator directly by calling the `base()` method.

## Return the Base Estimator
To return the base estimator:
```php
public base() : Estimator
```
