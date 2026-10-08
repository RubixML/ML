<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/Repeater.php">[source]</a></span>

# Repeater

Repeats the records of a data table a given number of times while they are in flight. Repeater re-traverses the base iterator for every pass rather than buffering the records in memory, which keeps memory usage constant even for very large or infinite streams.

!!! note
    The base iterator must be re-iterable. That means it has to be an array, an [IteratorAggregate](https://www.php.net/manual/en/class.iteratoraggregate.php) such as another extractor, a rewindable [Iterator](https://www.php.net/manual/en/class.iterator.php), or a callable factory that returns a fresh iterator for each pass. Raw [Generators](https://www.php.net/manual/en/class.generator.php) cannot be traversed twice by PHP and are therefore rejected when more than one pass is requested — pass the function that returns a fresh generator instead.

**Interfaces:** [Extractor](api.md)

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | iterator | | Traversable or callable | The base iterator or a callable that returns a fresh iterator for each pass. |
| 2 | repetitions | | int | The total number of passes over the records. |

## Example

```php
use Rubix\ML\Extractors\Repeater;
use Rubix\ML\Extractors\NDJSON;

$extractor = new Repeater(new NDJSON('example.jsonl'), 3);
```

## Generator Factory

A raw Generator can only be traversed once by PHP. To repeat one, pass the function that returns a fresh generator for each pass instead of the generator itself. Repeater invokes the factory at the beginning of every pass, so the records are still streamed in O(1) memory.

```php
use Rubix\ML\Extractors\Repeater;
use Generator;

function streamRecords() : Generator
{
    yield ['attitude' => 'nice', 'texture' => 'furry'];
    yield ['attitude' => 'mean', 'texture' => 'rough'];
}

$extractor = new Repeater(streamRecords(...), 3);

foreach ($extractor as $record) {
    // ...
}
```

## Additional Methods

This extractor does not have any additional methods.

## Streaming

Because the base iterator is re-traversed instead of buffered, Repeater streams each pass in O(1) memory. An infinite base iterator streams forever and the repetitions only begin once the base iterator is finished.

```php
use Rubix\ML\Extractors\Repeater;
use Rubix\ML\Extractors\CSV;

$extractor = new Repeater(new CSV('example.csv', true), 10);

foreach ($extractor as $record) {
    // ...
}
```
