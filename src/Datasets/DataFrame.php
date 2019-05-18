<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Other\Helpers\DataType;
use InvalidArgumentException;
use IteratorAggregate;
use RuntimeException;
use ArrayIterator;
use ArrayAccess;
use Countable;

class DataFrame implements ArrayAccess, IteratorAggregate, Countable
{
    /**
     * The rows of samples and columns of features that make up the
     * data table i.e. the fixed-length feature vectors.
     *
     * @var array[]
     */
    protected $samples;

    /**
     * @param array $samples
     * @param bool $validate
     * @throws \InvalidArgumentException
     */
    public function __construct(array $samples = [], bool $validate = true)
    {
        if ($validate) {
            $n = !empty($samples)
                ? is_array(reset($samples)) ? count(reset($samples)) : 1
                : 0;

            foreach ($samples as $index => &$sample) {
                $sample = is_array($sample)
                    ? array_values($sample)
                    : [$sample];

                if (count($sample) !== $n) {
                    throw new InvalidArgumentException('The number of feature'
                        . " columns must be equal for all samples, $n needed "
                        . count($sample) . ' given.');
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
     * Return the sample at the given row index.
     *
     * @param int $index
     * @return array
     */
    public function row(int $index) : array
    {
        return $this->offsetGet($index);
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
     * Return the feature column at the given index.
     *
     * @param int $index
     * @return array
     */
    public function column(int $index) : array
    {
        return array_column($this->samples, $index);
    }

    /**
     * Return the number of feature columns in the data frame.
     *
     * @return int
     */
    public function numColumns() : int
    {
        $sample = reset($this->samples);

        return $sample ? count($sample) : 0;
    }

    /**
     * Return an array of feature column data types autodectected using the first
     * sample in the dataframe.
     *
     * @return int[]
     */
    public function types() : array
    {
        return array_map([DataType::class, 'determine'], reset($this->samples) ?: []);
    }

    /**
     * Return the unique data types.
     *
     * @return int[]
     */
    public function uniqueTypes() : array
    {
        return array_unique($this->types());
    }

    /**
     * Does the dataframe consist of data of a single type?
     *
     * @return bool
     */
    public function homogeneous() : bool
    {
        return count($this->uniqueTypes()) === 1;
    }

    /**
     * Get the datatype for a feature column given a column index.
     *
     * @param int $index
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return int
     */
    public function columnType(int $index) : int
    {
        if (empty($this->samples)) {
            throw new RuntimeException('Cannot determine data type'
                . ' of an empty data frame.');
        }

        $sample = reset($this->samples);

        if (!isset($sample[$index])) {
            throw new InvalidArgumentException("Column $index does"
             . ' not exist.');
        }

        return DataType::determine($sample[$index]);
    }

    /**
     * Return a tuple containing the shape of the dataframe i.e the number of
     * rows and columns.
     *
     * @var int[]
     */
    public function shape() : array
    {
        return [$this->numRows(), $this->numColumns()];
    }

    /**
     * Return the number of elements in the dataframe.
     *
     * @return int
     */
    public function size() : int
    {
        return $this->numRows() * $this->numColumns();
    }

    /**
     * Rotate the dataframe and return it in an array. i.e. rows become
     * columns and columns become rows.
     *
     * @return array
     */
    public function columns() : array
    {
        if ($this->numRows() > 1) {
            return array_map(null, ...$this->samples);
        }

        $n = $this->numColumns();

        $columns = [];

        for ($i = 0; $i < $n; $i++) {
            $columns[] = array_column($this->samples, $i);
        }

        return $columns;
    }

    /**
     * Return the columns that match a given data type.
     *
     * @param int $type
     * @return array
     */
    public function columnsByType(int $type) : array
    {
        $n = $this->numColumns();

        $columns = [];

        for ($i = 0; $i < $n; $i++) {
            if ($this->columnType($i) === $type) {
                $columns[$i] = $this->column($i);
            }
        }

        return $columns;
    }

    /**
     * Is the dataframe empty?
     *
     * @return bool
     */
    public function empty() : bool
    {
        return empty($this->samples);
    }

    /**
     * @return int
     */
    public function count() : int
    {
        return $this->numRows();
    }

    /**
     * @param mixed $index
     * @param array $values
     * @throws \RuntimeException
     */
    public function offsetSet($index, $values) : void
    {
        throw new RuntimeException('Datasets cannot be mutated directly.');
    }

    /**
     * Does a given row exist in the dataframe.
     *
     * @param mixed $index
     * @return bool
     */
    public function offsetExists($index) : bool
    {
        return isset($this->samples[$index]);
    }

    /**
     * @param mixed $index
     * @throws \RuntimeException
     */
    public function offsetUnset($index) : void
    {
        throw new RuntimeException('Datasets cannot be mutated directly.');
    }

    /**
     * Return a column from the dataframe given by index.
     *
     * @param mixed $index
     * @throws \InvalidArgumentException
     * @return array
     */
    public function offsetGet($index) : array
    {
        return $this->samples[$index];
    }

    /**
     * Get an iterator for the samples in the dataset.
     *
     * @return \ArrayIterator
     */
    public function getIterator()
    {
        return new ArrayIterator($this->samples);
    }
}
