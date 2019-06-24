# Verbose
Verbose objects are capable of logging important events to any PSR-3 compatible logger such as [Monolog](https://github.com/Seldaek/monolog), [Analog](https://github.com/jbroadway/analog), or the included [Screen Logger](#screen). Logging is especially useful for monitoring the progress of the underlying learning algorithm in real time.

To set the logger pass in any PSR-3 compatible logger instance:
```php
public setLogger(LoggerInterface $logger) : void
```

### Example
```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('sentiment'));
```