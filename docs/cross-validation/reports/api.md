# Report Generators
Report generators output detailed reports from a validation set and a set of predictions. They are used in cross-validation to ascertain the generalization performance of an estimator.

### Generate a Report
To generate a report from the predictions of an estimator given the ground truth labels:
```php
public generate(array $predictions, array $labels) : Report
```

```php
use Rubix\ML\Reports\ConfusionMatrix;

$predictions = $estimator->predict($dataset);

$report = new ConfusionMatrix();

$results = $report->generate($predictions, $dataset->labels());
```

# Report Objects
The results of a report will be returned in a Report object whose attributes can be accessed like an associative array. In addition, report objects can be echoed to the terminal or even written to a file.

## Printing the Report
To display the human-readable form of the report, you can `echo` it out to the terminal.

```php
echo $results;
```

```sh
{
    "dog": {
        "dog": 12,
        "cat": 3,
        "turtle": 0
    },
    "cat": {
        "dog": 2,
        "cat": 9,
        "turtle": 1
    },
    "turtle": {
        "dog": 1,
        "cat": 0,
        "turtle": 11
    }
}
```

## Accessing Report Attributes
You can access individual report attributes by treating the report object as an associative array.

```php
$accuracy = $results['accuracy'];
```

## Encoding the Report
To return a JSON encoding that can be written to a file, call the `toJSON()` method on the report object.
```php
public toJSON(bool $pretty = true) : Encoding
```

```php
$encoding = $report->toJSON();
```
