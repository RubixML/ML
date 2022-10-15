<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Report;
use Rubix\ML\DataType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Extractors\Exporter;
use Rubix\ML\Transformers\Reversible;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use IteratorAggregate;
use ArrayAccess;
use Countable;

use function Rubix\ML\iterator_first;
use function Rubix\ML\iterator_filter;
use function Rubix\ML\array_transpose;
use function count;
use function is_array;

/**
 * Dataset
 *
 * In Rubix ML, data are passed in specialized in-memory containers called Dataset
 * objects. Dataset objects are extended table-like data structures with an internal
 * type system and many operations for wrangling. They can hold a heterogeneous mix
 * of categorical and continuous data and they make it easy to transport data in a
 * canonical way.
 *
 * > **Note:** By convention, categorical data are given as string type whereas
 * continuous data are given as either integer or floating point numbers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements ArrayAccess<int, array>
 * @implements IteratorAggregate<int, array>
 */
abstract class Dataset implements ArrayAccess, IteratorAggregate, Countable
{
    /**
     * The rows of samples and columns of features that make up the
     * data table i.e. the fixed-length feature vectors.
     *
     * @var list<list<mixed>>
     */
    protected array $samples;

    /**
     * @param mixed[] $samples
     * @param bool $verify
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(array $samples = [], bool $verify = true)
    {
        if ($samples and $verify) {
            $samples = array_values($samples);

            $prototype = array_values((array) current($samples));

            $n = count($prototype);

            $types = array_map([DataType::class, 'detect'], $prototype);

            foreach ($samples as $row => &$sample) {
                $sample = is_array($sample) ? array_values($sample) : [$sample];

                if (count($sample) !== $n) {
                    throw new InvalidArgumentException('Number of columns'
                        . " must be equal for all samples, $n expected but "
                        . count($sample) . " given at row offset $row.");
                }

                foreach ($sample as $column => $value) {
                    $type = DataType::detect($value);

                    if ($type != $types[$column]) {
                        throw new InvalidArgumentException("Column $column"
                            . ' must contain values of the same data type,'
                            . " $types[$column] expected but $type given at"
                            . " row offset $row.");
                    }
                }
            }
        }

        $this->samples = $samples;
    }

    /**
     * Build a dataset with the rows from an iterable data table.
     *
     * @param iterable<mixed[]> $iterator
     * @return static
     */
    abstract public static function fromIterator(iterable $iterator) : self;

    /**
     * Stack a number of datasets on top of each other to form a single dataset.
     *
     * @param iterable<\Rubix\ML\Datasets\Dataset> $datasets
     * @return static
     */
    abstract public static function stack(iterable $datasets) : self;

    /**
     * Return a 2-tuple containing the shape of the sample matrix i.e the number of rows and columns.
     *
     * @return array{int<0,max>,int<0,max>}
     */
    public function shape() : array
    {
        return [$this->numSamples(), $this->numFeatures()];
    }

    /**
     * Return the number of feature values in the dataset.
     *
     * @return int<0,max>
     */
    public function size() : int
    {
        return $this->numSamples() * $this->numFeatures();
    }

    /**
     * Return the high-level data types of each column in the data table.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function types() : array
    {
        $firstRow = iterator_first($this);

        return array_map([DataType::class, 'detect'], $firstRow);
    }

    /**
     * Return the number of rows in the datasets.
     *
     * @return int<0,max>
     */
    public function numSamples() : int
    {
        return count($this->samples);
    }

    /**
     * Return the sample at the given row offset.
     *
     * @param int $offset
     * @return list<mixed>
     */
    public function sample(int $offset) : array
    {
        if (isset($this->samples[$offset])) {
            return $this->samples[$offset];
        }

        throw new InvalidArgumentException("Sample at offset $offset not found.");
    }

    /**
     * Return the sample matrix.
     *
     * @return list<list<mixed>>
     */
    public function samples() : array
    {
        return $this->samples;
    }

    /**
     * Return the number of feature columns in the dataset.
     *
     * @return int<0,max>
     */
    public function numFeatures() : int
    {
        return isset($this->samples[0]) ? count($this->samples[0]) : 0;
    }

    /**
     * Return the feature column at the given offset.
     *
     * @param int $offset
     * @return mixed[]
     */
    public function feature(int $offset) : array
    {
        return array_column($this->samples, $offset);
    }

    /**
     * Drop a feature column at a given offset from the dataset.
     *
     * @param int $offset
     * @return self
     */
    public function dropFeature(int $offset) : self
    {
        foreach ($this->samples as &$sample) {
            array_splice($sample, $offset, 1);
        }

        return $this;
    }

    /**
     * Rotate the sample matrix so that the values of each feature become rows.
     *
     * @return mixed[]
     */
    public function features() : array
    {
        return array_transpose($this->samples);
    }

    /**
     * Return the feature columns that match a given data type.
     *
     * @param \Rubix\ML\DataType $type
     * @return mixed[]
     */
    public function featuresByType(DataType $type) : array
    {
        $columns = [];

        foreach ($this->featureTypes() as $offset => $featureType) {
            if ($featureType == $type) {
                $columns[$offset] = $this->feature($offset);
            }
        }

        return $columns;
    }

    /**
     * Get the data type for a feature column at the given offset.
     *
     * @param int $offset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\DataType
     */
    public function featureType(int $offset) : DataType
    {
        if (empty($this->samples)) {
            throw new RuntimeException('Cannot determine data type of empty dataset.');
        }

        $prototype = $this->samples[0];

        if (!isset($prototype[$offset])) {
            throw new InvalidArgumentException('Column at offset'
                . " $offset does not exist.");
        }

        return DataType::detect($prototype[$offset]);
    }

    /**
     * Return an array of feature column data types autodetected using the first sample in the dataset.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function featureTypes() : array
    {
        if (empty($this->samples)) {
            throw new RuntimeException('Cannot determine data types of empty dataset.');
        }

        return array_map([DataType::class, 'detect'], $this->samples[0] ?? []);
    }

    /**
     * Return the unique feature types.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function uniqueTypes() : array
    {
        return array_unique($this->featureTypes());
    }

    /**
     * Do the samples consist of values of a single data type?
     *
     * @return bool
     */
    public function homogeneous() : bool
    {
        return count($this->uniqueTypes()) === 1;
    }

    /**
     * Apply a transformation to the dataset.
     *
     * @param \Rubix\ML\Transformers\Transformer $transformer
     * @return static
     */
    public function apply(Transformer $transformer) : self
    {
        if ($transformer instanceof Stateful) {
            if (!$transformer->fitted()) {
                $transformer->fit($this);
            }
        }

        $transformer->transform($this->samples);

        return $this;
    }

    /**
     * Reverse a transformation that was applied to the dataset.
     *
     * @param \Rubix\ML\Transformers\Reversible $transformer
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return static
     */
    public function reverseApply(Reversible $transformer) : self
    {
        if ($transformer instanceof Stateful) {
            if (!$transformer->fitted()) {
                throw new RuntimeException('Stateful transformer has not been fitted.');
            }
        }

        $transformer->reverseTransform($this->samples);

        return $this;
    }

    /**
     * Filter the records of the dataset using a callback function to determine if a row should be included in the return dataset.
     *
     * @param callable $callback
     * @return static
     */
    public function filter(callable $callback) : self
    {
        return static::fromIterator(iterator_filter($this, $callback));
    }

    /**
     * Return an array of statistics such as the central tendency, dispersion
     * and shape of each continuous feature column and the joint probabilities
     * of every categorical feature column.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Report
     */
    public function describe() : Report
    {
        if ($this->empty()) {
            throw new RuntimeException('Cannot describe an empty dataset.');
        }

        $columns = array_transpose(iterator_to_array($this));

        $stats = [];

        foreach ($this->types() as $offset => $type) {
            $description = [
                'offset' => $offset,
                'type' => (string) $type,
            ];

            $values = $columns[$offset];

            switch ($type->code()) {
                case DataType::CONTINUOUS:
                    [$mean, $variance] = Stats::meanVar($values);

                    [$min, $p25, $median, $p75, $max] = Stats::quantiles($values, [
                        0.0, 0.25, 0.5, 0.75, 1.0,
                    ]);

                    $description += [
                        'mean' => $mean,
                        'variance' => $variance,
                        'standard deviation' => sqrt($variance),
                        'skewness' => Stats::skewness($values, $mean),
                        'kurtosis' => Stats::kurtosis($values, $mean),
                        'min' => $min,
                        '25%' => $p25,
                        'median' => $median,
                        '75%' => $p75,
                        'max' => $max,
                        'range' => $max - $min,
                    ];

                    break;

                case DataType::CATEGORICAL:
                    $counts = array_count_values($values);

                    $total = count($values);

                    $probabilities = [];

                    foreach ($counts as $category => $count) {
                        $probabilities[$category] = $count / $total;
                    }

                    arsort($probabilities);

                    $description += [
                        'num categories' => count($probabilities),
                        'probabilities' => $probabilities,
                    ];

                    break;
            }

            $stats[] = $description;
        }

        return new Report($stats);
    }

    /**
     * Sort the records in the dataset using a callback for comparisons between samples. The callback function
     * accepts two records to be compared and should return `true` if the records should be swapped.
     *
     * @param callable $callback
     * @return static
     */
    public function sort(callable $callback) : self
    {
        $records = iterator_to_array($this);

        $nHat = count($records) - 1;

        for ($i = 0; $i < $nHat; ++$i) {
            $swapped = false;

            for ($j = 0; $j < $nHat - $i; ++$j) {
                $recordA = $records[$j];
                $recordB = $records[$j + 1];

                if ($callback($recordA, $recordB)) {
                    $records[$j] = $recordB;
                    $records[$j + 1] = $recordA;

                    $swapped = true;
                }
            }

            if (!$swapped) {
                break;
            }
        }

        return static::fromIterator($records);
    }

    /**
     * Remove duplicate rows from the dataset.
     *
     * @return self
     */
    public function deduplicate() : self
    {
        return static::fromIterator(array_unique(iterator_to_array($this), SORT_REGULAR));
    }

    /**
     * Write the dataset to the location and format given by a writable extractor.
     *
     * @param \Rubix\ML\Extractors\Exporter $extractor
     */
    public function exportTo(Exporter $extractor) : void
    {
        $extractor->export($this);
    }

    /**
     * Is the dataset empty?
     *
     * @return bool
     */
    public function empty() : bool
    {
        return empty($this->samples);
    }

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param int $n
     * @return static
     */
    abstract public function head(int $n = 10) : self;

    /**
     * Return a dataset containing only the last n samples.
     *
     * @param int $n
     * @return static
     */
    abstract public function tail(int $n = 10) : self;

    /**
     * Take n samples from the dataset and return them in a new dataset.
     *
     * @param int $n
     * @return static
     */
    abstract public function take(int $n = 1) : self;

    /**
     * Leave n samples on the dataset and return the rest in a new dataset.
     *
     * @param int $n
     * @return static
     */
    abstract public function leave(int $n = 1) : self;

    /**
     * Return an n size portion of the dataset in a new dataset.
     *
     * @param int $offset
     * @param int $n
     * @return static
     */
    abstract public function slice(int $offset, int $n) : self;

    /**
     * Remove a size n chunk of the dataset starting at offset and return it in
     * a new dataset.
     *
     * @param int $offset
     * @param int $n
     * @return static
     */
    abstract public function splice(int $offset, int $n) : self;

    /**
     * Merge another dataset with this dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return static
     */
    abstract public function merge(Dataset $dataset) : self;

    /**
     * Join the columns of this dataset with another dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return static
     */
    abstract public function join(Dataset $dataset) : self;

    /**
     * Split the dataset into two subsets with a given ratio of samples.
     *
     * @param float $ratio
     * @return array{self,self}
     */
    abstract public function split(float $ratio = 0.5) : array;

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param int $k
     * @return list<self>
     */
    abstract public function fold(int $k = 10) : array;

    /**
     * Generate a collection of batches of size n from the dataset. If there are
     * not enough samples to fill an entire batch, then the dataset will contain
     * as many samples as possible.
     *
     * @param int $n
     * @return list<self>
     */
    abstract public function batch(int $n = 50) : array;

    /**
     * Partition the dataset into left and right subsets using the values of a single feature column for comparison.
     *
     * @internal
     *
     * @param int $offset
     * @param mixed $value
     * @return array{self,self}
     */
    abstract public function splitByFeature(int $offset, $value) : array;

    /**
     * Partition the dataset into left and right subsets based on the samples' distances from two centroids.
     *
     * @internal
     *
     * @param (string|int|float)[] $leftCentroid
     * @param (string|int|float)[] $rightCentroid
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return array{self,self}
     */
    abstract public function spatialSplit(array $leftCentroid, array $rightCentroid, Distance $kernel);

    /**
     * Randomize the dataset.
     *
     * @return static
     */
    abstract public function randomize() : self;

    /**
     * Generate a random subset without replacement.
     *
     * @param int $n
     * @return self
     */
    abstract public function randomSubset(int $n);

    /**
     * Generate a random subset of n samples with replacement.
     *
     * @param int $n
     * @return self
     */
    abstract public function randomSubsetWithReplacement(int $n);

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param int $n
     * @param (int|float)[] $weights
     * @return self
     */
    abstract public function randomWeightedSubsetWithReplacement(int $n, array $weights);

    /**
     * Return the number of rows in the dataset.
     *
     * @return int
     */
    public function count() : int
    {
        return $this->numSamples();
    }

    /**
     * @param int $offset
     * @param mixed[] $values
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function offsetSet($offset, $values) : void
    {
        throw new RuntimeException('Datasets cannot be mutated directly.');
    }

    /**
     * Does a given row exist in the dataset.
     *
     * @param int $offset
     * @return bool
     */
    public function offsetExists($offset) : bool
    {
        return isset($this->samples[$offset]);
    }

    /**
     * @param int $offset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function offsetUnset($offset) : void
    {
        throw new RuntimeException('Datasets cannot be mutated directly.');
    }
}
