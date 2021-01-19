<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Other\Helpers\JSON;
use Rubix\ML\Report;
use Rubix\ML\DataType;
use Rubix\ML\Encoding;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Exceptions\InvalidArgumentException;
use IteratorAggregate;
use JsonSerializable;
use Rubix\ML\Exceptions\RuntimeException;
use ArrayAccess;
use Stringable;
use Countable;

use function Rubix\ML\array_transpose;
use function count;
use function is_array;

use const Rubix\ML\EPSILON;

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
abstract class Dataset implements ArrayAccess, IteratorAggregate, JsonSerializable, Countable, Stringable
{
    /**
     * The rows of samples and columns of features that make up the
     * data table i.e. the fixed-length feature vectors.
     *
     * @var array[]
     */
    protected $samples;

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
     * @param iterable<array> $iterator
     * @return self
     */
    abstract public static function fromIterator(iterable $iterator);

    /**
     * Stack a number of datasets on top of each other to form a single
     * dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset[] $datasets
     * @return self
     */
    abstract public static function stack(array $datasets);

    /**
     * Return the sample matrix.
     *
     * @return array[]
     */
    public function samples() : array
    {
        return $this->samples;
    }

    /**
     * Return the sample at the given row offset.
     *
     * @param int $offset
     * @return mixed[]
     */
    public function sample(int $offset) : array
    {
        if (isset($this->samples[$offset])) {
            return $this->samples[$offset];
        }

        throw new InvalidArgumentException("Sample at offset $offset not found.");
    }

    /**
     * Return the number of rows in the datasets.
     *
     * @return int
     */
    public function numRows() : int
    {
        return count($this->samples);
    }

    /**
     * Return the feature column at the given offset.
     *
     * @param int $offset
     * @return mixed[]
     */
    public function column(int $offset) : array
    {
        return array_column($this->samples, $offset);
    }

    /**
     * Return the number of feature columns in the dataset.
     *
     * @return int
     */
    public function numColumns() : int
    {
        return isset($this->samples[0]) ? count($this->samples[0]) : 0;
    }

    /**
     * Return an array of feature column data types autodectected using the first
     * sample in the dataset.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function columnTypes() : array
    {
        return array_map([DataType::class, 'detect'], $this->samples[0] ?? []);
    }

    /**
     * Return the unique data types.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function uniqueTypes() : array
    {
        return array_unique($this->columnTypes());
    }

    /**
     * Does the dataset consist of data of a single type?
     *
     * @return bool
     */
    public function homogeneous() : bool
    {
        return count($this->uniqueTypes()) === 1;
    }

    /**
     * Get the datatype for a feature column at the given offset.
     *
     * @param int $offset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\DataType
     */
    public function columnType(int $offset) : DataType
    {
        if (empty($this->samples)) {
            throw new RuntimeException('Cannot determine data type'
                . ' of an empty dataset.');
        }

        if (!isset($this->samples[0][$offset])) {
            throw new InvalidArgumentException('Column at offset'
                . " $offset does not exist.");
        }

        return DataType::detect($this->samples[0][$offset]);
    }

    /**
     * Return a 2-tuple containing the shape of the dataset i.e the number of
     * rows and columns.
     *
     * @return int[]
     */
    public function shape() : array
    {
        return [$this->numRows(), $this->numColumns()];
    }

    /**
     * Return the number of elements in the dataset.
     *
     * @return int
     */
    public function size() : int
    {
        return $this->numRows() * $this->numColumns();
    }

    /**
     * Rotate the dataset and return it in an array. i.e. rows become
     * columns and columns become rows.
     *
     * @return array[]
     */
    public function columns() : array
    {
        return array_transpose($this->samples);
    }

    /**
     * Return the columns that match a given data type.
     *
     * @param \Rubix\ML\DataType $type
     * @return array[]
     */
    public function columnsByType(DataType $type) : array
    {
        $columns = [];

        foreach ($this->columnTypes() as $offset => $columnType) {
            if ($columnType == $type) {
                $columns[$offset] = $this->column($offset);
            }
        }

        return $columns;
    }

    /**
     * Transform a feature column with a callback function.
     *
     * @param int $offset
     * @param callable $callback
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function transformColumn(int $offset, callable $callback) : self
    {
        if ($offset < 0 or $offset >= $this->numColumns()) {
            throw new InvalidArgumentException('Column offset must'
                . " be between 0 and {$this->numColumns()}, $offset"
                . ' given.');
        }

        foreach ($this->samples as &$sample) {
            $value = &$sample[$offset];

            $value = $callback($value);
        }

        return $this;
    }

    /**
     * Drop the column at the given offset.
     *
     * @param int $offset
     * @return self
     */
    public function dropColumn(int $offset) : self
    {
        return $this->dropColumns([$offset]);
    }

    /**
     * Drop the columns at the given offsets.
     *
     * @param int[] $offsets
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function dropColumns(array $offsets) : self
    {
        if (empty($offsets)) {
            return $this;
        }

        foreach ($this->samples as &$sample) {
            foreach ($offsets as $offset) {
                unset($sample[$offset]);
            }

            $sample = array_values($sample);
        }

        return $this;
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
     * Return a JSON representation of the dataset.
     *
     * @param bool $pretty
     * @return \Rubix\ML\Encoding
     */
    public function toJSON(bool $pretty = false) : Encoding
    {
        return new Encoding(JSON::encode($this, $pretty ? JSON_PRETTY_PRINT : 0) ?: '');
    }

    /**
     * Return a newline delimited JSON representation of the dataset.
     *
     * @param string[]|null $header
     * @return \Rubix\ML\Encoding
     */
    public function toNDJSON(?array $header = null) : Encoding
    {
        if ($header) {
            $cols = $this->numColumns()
                + ($this instanceof Labeled ? 1 : 0);

            if (count($header) !== $cols) {
                throw new InvalidArgumentException('Header must have'
                    . " $cols columns, " . count($header) . ' given.');
            }
        }

        $ndjson = '';

        foreach ($this as $row) {
            if ($header) {
                $row = array_combine($header, $row);
            }

            $ndjson .= JSON::encode($row) . PHP_EOL;
        }

        return new Encoding($ndjson);
    }

    /**
     * Return the dataset as comma-separated values (CSV) string.
     *
     * @param string[]|null $header
     * @param string $delimiter
     * @param string $enclosure
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return \Rubix\ML\Encoding
     */
    public function toCSV(?array $header = null, string $delimiter = ',', string $enclosure = '"') : Encoding
    {
        if ($header) {
            $cols = $this->numColumns()
                + ($this instanceof Labeled ? 1 : 0);

            if (count($header) !== $cols) {
                throw new InvalidArgumentException('Header must have'
                    . " $cols columns, " . count($header) . ' given.');
            }
        }

        if (empty($delimiter)) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' at least 1 character.');
        }

        if (strlen($enclosure) !== 1) {
            throw new InvalidArgumentException('Enclosure must be'
                . ' a single character.');
        }

        $csv = '';

        if ($header) {
            foreach ($header as $title) {
                if (str_contains($title, $delimiter)) {
                    $title = $enclosure . $title . $enclosure;
                }
            }

            $csv .= implode($delimiter, $header) . PHP_EOL;
        }

        foreach ($this as $row) {
            foreach ($row as &$value) {
                if (str_contains($value, $delimiter)) {
                    $value = $enclosure . $value . $enclosure;
                }
            }

            $csv .= implode($delimiter, $row) . PHP_EOL;
        }

        return new Encoding($csv);
    }

    /**
     * Return an array of statistics such as the central tendency, dispersion
     * and shape of each continuous feature column and the joint probabilities
     * of every categorical feature column.
     *
     * @return \Rubix\ML\Report
     */
    public function describe() : Report
    {
        $stats = [];

        foreach ($this->columnTypes() as $offset => $type) {
            $desc = [
                'column' => $offset,
                'type' => (string) $type,
            ];

            switch ($type->code()) {
                case DataType::CONTINUOUS:
                    $values = $this->column($offset);

                    [$mean, $variance] = Stats::meanVar($values);

                    $quartiles = Stats::quantiles($values, [
                        0.0, 0.25, 0.5, 0.75, 1.0,
                    ]);

                    $desc += [
                        'mean' => $mean,
                        'variance' => $variance,
                        'std_dev' => sqrt($variance ?: EPSILON),
                        'skewness' => Stats::skewness($values, $mean),
                        'kurtosis' => Stats::kurtosis($values, $mean),
                        'min' => $quartiles[0],
                        '25%' => $quartiles[1],
                        'median' => $quartiles[2],
                        '75%' => $quartiles[3],
                        'max' => $quartiles[4],
                    ];

                    break;

                case DataType::CATEGORICAL:
                    $values = $this->column($offset);

                    $counts = array_count_values($values);

                    $total = count($values);

                    $probabilities = [];

                    foreach ($counts as $category => $count) {
                        $probabilities[$category] = $count / $total;
                    }

                    arsort($probabilities);

                    $desc += [
                        'num_categories' => count($probabilities),
                        'probabilities' => $probabilities,
                    ];

                    break;
            }

            $stats[] = $desc;
        }

        return new Report($stats);
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
     * @return self
     */
    abstract public function head(int $n = 10);

    /**
     * Return a dataset containing only the last n samples.
     *
     * @param int $n
     * @return self
     */
    abstract public function tail(int $n = 10);

    /**
     * Take n samples from the dataset and return them in a new dataset.
     *
     * @param int $n
     * @return self
     */
    abstract public function take(int $n = 1);

    /**
     * Leave n samples on the dataset and return the rest in a new dataset.
     *
     * @param int $n
     * @return self
     */
    abstract public function leave(int $n = 1);

    /**
     * Return an n size portion of the dataset in a new dataset.
     *
     * @param int $offset
     * @param int $n
     * @return self
     */
    abstract public function slice(int $offset, int $n);

    /**
     * Remove a size n chunk of the dataset starting at offset and return it in
     * a new dataset.
     *
     * @param int $offset
     * @param int $n
     * @return self
     */
    abstract public function splice(int $offset, int $n);

    /**
     * Merge another dataset with this dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    abstract public function merge(Dataset $dataset);

    /**
     * Join the columns of this dataset with another dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    abstract public function join(Dataset $dataset);

    /**
     * Merge the columns of this dataset with another dataset.
     *
     * @deprecated
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    abstract public function augment(Dataset $dataset);

    /**
     * Drop the row at the given offset.
     *
     * @param int $offset
     * @return self
     */
    abstract public function dropRow(int $offset);

    /**
     * Drop the rows at the given indices.
     *
     * @param int[] $indices
     * @return self
     */
    abstract public function dropRows(array $indices);

    /**
     * Randomize the dataset.
     *
     * @return self
     */
    abstract public function randomize();

    /**
     * Filter the rows of the dataset using the values of a feature column as the
     * argument to a callback.
     *
     * @param int $offset
     * @param callable $fn
     * @return self
     */
    abstract public function filterByColumn(int $offset, callable $fn);

    /**
     * Sort the dataset by a column in the sample matrix.
     *
     * @param int $offset
     * @param bool $descending
     * @return self
     */
    abstract public function sortByColumn(int $offset, bool $descending = false);

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
    abstract public function splitByColumn(int $offset, $value) : array;

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
     * Remove duplicate rows from the dataset.
     *
     * @return self
     */
    abstract public function deduplicate();

    /**
     * Return the dataset object as a data table array.
     *
     * @return array[]
     */
    abstract public function toArray() : array;

    /**
     * @return array[]
     */
    public function jsonSerialize() : array
    {
        return $this->toArray();
    }

    /**
     * Return the number of rows in the dataset.
     *
     * @return int
     */
    public function count() : int
    {
        return $this->numRows();
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

    /**
     * Return a string representation of the first few rows of the dataset.
     *
     * @return string
     */
    abstract public function __toString() : string;
}
