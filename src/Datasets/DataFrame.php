<?php

namespace Rubix\ML\Datasets;

use InvalidArgumentException;
use IteratorAggregate;
use RuntimeException;
use ArrayIterator;
use ArrayAccess;
use Countable;

class DataFrame implements ArrayAccess, IteratorAggregate, Countable
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;
    const RESOURCE = 3;

    /**
     * The feature vectors of the dataset. i.e the data table.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * @param  array  $samples
     * @param  bool  $validate
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $samples = [], bool $validate = true)
    {
        if ($validate) {
            $samples = array_values($samples);

            $n = is_array(current($samples)) ? count(current($samples)) : 1;
            $n = empty($samples) ? 0 : $n;

            foreach ($samples as &$sample) {
                if (is_array($sample)) {
                    $sample = array_values($sample);
                } else {
                    $sample = [$sample];
                }

                if (count($sample) !== $n) {
                    throw new InvalidArgumentException('The number of feature'
                        . " columns must be equal for all samples, $n needed "
                        . count($sample) . ' given.');
                }

                foreach ($sample as $feature) {
                    if (is_string($feature) or is_numeric($feature) or is_resource($feature)) {
                        continue 1;
                    }

                    throw new InvalidArgumentException('Feature must be a'
                        . ' resource, string or numeric type, '
                        . gettype($feature) . ' found.');
                }
            }
        }

        $this->samples = $samples;
    }

    /**
     * Return the sample matrix.
     *
     * @return array
     */
    public function samples() : array
    {
        return $this->samples;
    }

    /**
     * Return the sample at the given row index.
     *
     * @param  int  $index
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
     * @param  int  $index
     * @throws \InvalidArgumentException
     * @return array
     */
    public function column(int $index) : array
    {
        if (!isset($this->samples[0][$index])) {
            throw new InvalidArgumentException("Column $index does not exist.");
        }

        return array_column($this->samples, $index);
    }

    /**
     * Return the number of feature columns in the data frame.
     *
     * @return int
     */
    public function numColumns() : int
    {
        return count($this->samples[0] ?? []);
    }

    /**
     * Return an array of feature column datatypes autodectected using the first
     * sample in the dataframe.
     *
     * @return array
     */
    public function types() : array
    {
        $types = [];

        for ($i = 0; $i < $this->numColumns(); $i++) {
            $types[] = $this->columnType($i);
        }

        return $types;
    }

    /**
     * Return the number of columns match a specific data type.
     * 
     * @param  int  $type
     * @return int
     */
    public function typeCount(int $type) : int
    {
        $counts = $this->typeCounts();

        return $counts[$type] ?? 0;
    }

    /**
     * Return the number of feature columns for each data type.
     * 
     * @return int[]
     */
    public function typeCounts() : array
    {
        $counts = [
            self::CATEGORICAL => 0,
            self::CONTINUOUS => 0,
            self::RESOURCE => 0,
        ];

        return array_replace($counts, array_count_values($this->types()));
    }

    /**
     * Get the datatype for a feature column given a column index.
     *
     * @param  int  $index
     * @throws \InvalidArgumentException
     * @return int|null
     */
    public function columnType(int $index) : ?int
    {
        if (!isset($this->samples[0])) {
            return null;
        }

        if (!isset($this->samples[0][$index])) {
            throw new InvalidArgumentException("Column $index does not exist.");
        }

        $feature = $this->samples[0][$index];

        switch (true) {
            case is_string($feature):
                return self::CATEGORICAL;

            case is_numeric($feature):
                return self::CONTINUOUS;

            case is_resource($feature):
                return self::RESOURCE;

            default:
                return null;
        }
    }

    /**
     * Return a tuple containing the shape of the dataframe i.e the number of
     * rows and columns.
     *
     * @var array
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
     * @param  int  $type
     * @return array
     */
    public function columnsByType(int $type) : array
    {
        $n = $this->numColumns();

        $columns = [];

        for ($i = 0; $i < $n; $i++) {
            if ($this->columnType($i) === $type) {
                $columns[$i] = array_column($this->samples, $i);
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
     * @param  mixed  $index
     * @param  array  $values
     * @throws \RuntimeException
     * @return void
     */
    public function offsetSet($index, $values) : void
    {
        throw new RuntimeException('Datasets cannot be mutated directly.');
    }

    /**
     * Does a given column exist in the dataset.
     *
     * @param  mixed  $index
     * @return bool
     */
    public function offsetExists($index) : bool
    {
        return isset($this->samples[$index]);
    }

    /**
     * @param  mixed  $index
     * @throws \RuntimeException
     * @return void
     */
    public function offsetUnset($index) : void
    {
        throw new RuntimeException('Datasets cannot be mutated directly.');
    }

    /**
     * Return a column from the dataframe given by index.
     *
     * @param  mixed  $index
     * @throws \InvalidArgumentException
     * @return array
     */
    public function offsetGet($index) : array
    {
        if (!isset($this->samples[$index])) {
            throw new InvalidArgumentException('Sample not found at index'
                . (string) $index . '.');
        }

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
