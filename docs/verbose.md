# Verbose
Verbose objects are capable of logging important events to any PSR-3 compatible logger such as [Monolog](https://github.com/Seldaek/monolog), [Analog](https://github.com/jbroadway/analog), or the included [Screen Logger](other/loggers/screen.md). Logging is especially useful for monitoring the progress of the underlying learning algorithm in real-time.

## Set the Logger
To set the logger pass in any PSR-3 compatible logger instance:
```php
public setLogger(LoggerInterface $logger) : void
```

## Return the Logger
Return the logger or null if not set:
```php
public logger() : ?LoggerInterface
```

```php
use Rubix\ML\Regressors\Adaline;
use Rubix\ML\Other\Loggers\Screen;

$estimator = new Adaline();

$estimator->setLogger(new Screen('example'));

$estimator->train($dataset);
```

```sh
[2020-08-05 04:26:11] INFO: Learner init Adaline {batch_size: 128, optimizer: Adam {rate: 0.01, momentum_decay: 0.1, norm_decay: 0.001}, alpha: 0.0001, epochs: 100, min_change: 0.001, window: 5, cost_fn: Huber Loss {alpha: 1}}
[2020-08-05 04:26:11] INFO: Training started
[2020-08-05 04:26:11] example.INFO: Epoch 1 - Huber Loss {alpha: 1}: 0.36839299586132
[2020-08-05 04:26:11] example.INFO: Epoch 2 - Huber Loss {alpha: 1}: 0.0018235958273629
[2020-08-05 04:26:11] example.INFO: Epoch 3 - Huber Loss {alpha: 1}: 0.0017358090553563
[2020-08-05 04:26:11] example.INFO: Training complete
```
