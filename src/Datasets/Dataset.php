<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\DataType;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;
use IteratorAggregate;
use JsonSerializable;
use RuntimeException;
use ArrayAccess;
use Countable;

use function Rubix\ML\array_transpose;
use function count;

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
abstract class Dataset implements ArrayAccess, IteratorAggregate, JsonSerializable, Countable
{
    /**
     * The rows of samples and columns of features that make up the
     * data table i.e. the fixed-length feature vectors.
     *
     * @var array[]
     */
    protected $samples;

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
     * @param array[] $samples
     * @param bool $validate
     * @throws \InvalidArgumentException
     */
    public function __construct(array $samples = [], bool $validate = true)
    {
        if ($validate and $samples) {
            $samples = array_values($samples);

            $proto = isset($samples[0]) ? array_values($samples[0]) : [];

            $n = count($proto);

            $types = array_map([DataType::class, 'determine'], $proto);

            foreach ($samples as $row => &$sample) {
                $sample = array_values($sample);

                if (count($sample) !== $n) {
                    throw new InvalidArgumentException('Number of columns'
                        . " must be equal for all samples, $n expected but"
                        . ' ' . count($sample) . ' given.');
                }

                foreach ($sample as $column => $value) {
                    if (DataType::determine($value) != $types[$column]) {
                        throw new InvalidArgumentException("Column $column must"
                            . ' contain values of the same data type,'
                            . " $types[$column] expected but "
                            . DataType::determine($value) . ' given.');
                    }
                }
            }
        }

        $this->samples = $samples;
    }

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
     * @return \Rubix\ML\DataType[]
     */
    public function columnTypes() : array
    {
        return array_map([DataType::class, 'determine'], $this->samples[0] ?? []);
    }

    /**
     * Return the unique data types.
     *
     * @return \Rubix\ML\DataType[]
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
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
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

        return DataType::determine($this->samples[0][$offset]);
    }

    /**
     * Return a tuple containing the shape of the dataset i.e the number of
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
        $n = $this->numColumns();

        $columns = [];

        for ($i = 0; $i < $n; ++$i) {
            if ($this->columnType($i) == $type) {
                $columns[$i] = $this->column($i);
            }
        }

        return $columns;
    }

    /**
     * Transform a feature column with a callback function.
     *
     * @param int $column
     * @param callable $callback
     * @throws \InvalidArgumentException
     * @return self
     */
    public function transformColumn(int $column, callable $callback) : self
    {
        if ($column < 0 or $column >= $this->numColumns()) {
            throw new InvalidArgumentException('Column number must'
                . " be between 0 and {$this->numColumns()}, $column"
                . ' given.');
        }

        foreach ($this->samples as &$sample) {
            $value = &$sample[$column];

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
     * @throws \InvalidArgumentException
     * @return self
     */
    public function dropColumns(array $offsets) : self
    {
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
     * @return self
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
     * Return an array of statistics such as the central tendency, dispersion
     * and shape of each continuous feature column and the joint probabilities
     * of every categorical feature column.
     *
     * @return mixed[]
     */
    public function describe() : array
    {
        $stats = [];

        foreach ($this->columnTypes() as $column => $type) {
            $desc = [];

            $desc['type'] = (string) $type;

            switch ($type->code()) {
                case DataType::CONTINUOUS:
                    $values = $this->column($column);

                    [$mean, $variance] = Stats::meanVar($values);

                    $desc['mean'] = $mean;
                    $desc['variance'] = $variance;
                    $desc['std_dev'] = sqrt($variance ?: EPSILON);
                    $desc['skewness'] = Stats::skewness($values, $mean);
                    $desc['kurtosis'] = Stats::kurtosis($values, $mean);

                    $quartiles = Stats::quantiles($values, [
                        0.0, 0.25, 0.5, 0.75, 1.0,
                    ]);

                    $desc['min'] = $quartiles[0];
                    $desc['25%'] = $quartiles[1];
                    $desc['median'] = $quartiles[2];
                    $desc['75%'] = $quartiles[3];
                    $desc['max'] = $quartiles[4];

                    break 1;

                case DataType::CATEGORICAL:
                    $values = $this->column($column);
                    
                    $counts = array_count_values($values);

                    $total = count($values) ?: EPSILON;

                    $probabilities = [];

                    foreach ($counts as $category => $count) {
                        $probabilities[$category] = $count / $total;
                    }

                    $desc['num_categories'] = count($counts);
                    $desc['probabilities'] = $probabilities;

                    break 1;
            }

            $stats[$column] = $desc;
        }

        return $stats;
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
     * Return a JSON representation of the dataset.
     *
     * @param bool $pretty
     * @return string
     */
    public function toJSON(bool $pretty = false) : string
    {
        return json_encode($this, $pretty ? JSON_PRETTY_PRINT : 0) ?: '';
    }

    /**
     * Return a newline delimited JSON representation of the dataset.
     *
     * @return string
     */
    public function toNDJSON() : string
    {
        $ndjson = '';

        foreach ($this->getIterator() as $row) {
            $ndjson .= json_encode($row) . PHP_EOL;
        }

        return $ndjson;
    }

    /**
     * Return the dataset as comma-separated values (CSV) string.
     *
     * @param string $delimiter
     * @param string $enclosure
     * @throws \InvalidArgumentException
     * @return string
     */
    public function toCSV(string $delimiter = ',', string $enclosure = '"') : string
    {
        if (strlen($delimiter) !== 1) {
            throw new InvalidArgumentException('Delimiter must be'
                . ' a single character.');
        }

        if (strlen($enclosure) !== 1) {
            throw new InvalidArgumentException('Enclosure must be'
                . ' a single character.');
        }
        
        $csv = '';

        foreach ($this->getIterator() as $row) {
            foreach ($row as &$value) {
                $value = (string) $value;

                if (strpos($value, $delimiter) !== false) {
                    $value = $enclosure . $value . $enclosure;
                }
            }

            $csv .= implode($delimiter, $row) . PHP_EOL;
        }

        return $csv;
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
     * Merge the columns of this dataset with another dataset.
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
     * @return self[]
     */
    abstract public function split(float $ratio = 0.5) : array;

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param int $k
     * @return self[]
     */
    abstract public function fold(int $k = 10) : array;

    /**
     * Generate a collection of batches of size n from the dataset. If there are
     * not enough samples to fill an entire batch, then the dataset will contain
     * as many samples as possible.
     *
     * @param int $n
     * @return self[]
     */
    abstract public function batch(int $n = 50) : array;

    /**
     * Partition the dataset into left and right subsets using the values of a single
     * feature column for comparison.
     *
     * @param int $offset
     * @param mixed $value
     * @return self[]
     */
    abstract public function partitionByColumn(int $offset, $value) : array;

    /**
     * Partition the dataset into left and right subsets based on the samples' distances
     * from two centroids.
     *
     * @param (string|int|float)[] $leftCentroid
     * @param (string|int|float)[] $rightCentroid
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @return self[]
     */
    abstract public function spatialPartition(array $leftCentroid, array $rightCentroid, ?Distance $kernel = null);

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
     * @param mixed $offset
     * @param mixed[] $values
     * @throws \RuntimeException
     */
    public function offsetSet($offset, $values) : void
    {
        throw new RuntimeException('Datasets cannot be mutated directly.');
    }

    /**
     * Does a given row exist in the dataset.
     *
     * @param mixed $offset
     * @return bool
     */
    public function offsetExists($offset) : bool
    {
        return isset($this->samples[$offset]);
    }

    /**
     * @param mixed $offset
     * @throws \RuntimeException
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
