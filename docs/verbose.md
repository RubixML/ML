# Verbose
Verbose Learners are capable of logging important events to any PSR-3 compatible logger such as [Monolog](https://github.com/Seldaek/monolog), [Analog](https://github.com/jbroadway/analog), or the included [Screen Logger](other/loggers/screen.md). Logging is especially useful for monitoring the progress of the underlying learning algorithm in real-time.

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

**Example**

```php
use Rubix\ML\Regressors\Adaline;
use Rubix\ML\Other\Loggers\Screen;

$estimator = new Adaline();

$estimator->setLogger(new Screen('example'));

$logger = $estimator->logger();
```