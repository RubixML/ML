<?php

namespace Rubix\ML\Other\Structures;

use InvalidArgumentException;
use IteratorAggregate;
use RuntimeException;
use ArrayIterator;
use ArrayAccess;
use Countable;

class DataFrame implements ArrayAccess, IteratorAggregate, Countable
{
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
        if ($validate === true) {
            $samples = array_values($samples);

            $numFeatures = isset($samples[0]) ? count($samples[0]) : 0;

            foreach ($samples as &$sample) {
                if (count($sample) !== $numFeatures) {
                    throw new InvalidArgumentException('The number of feature'
                        . ' columns must be equal for all samples.');
                }

                $sample = array_values($sample);

                foreach ($sample as $feature) {
                    if (!is_string($feature) and !is_numeric($feature)) {
                        throw new InvalidArgumentException('Feature must be a'
                            . ' string, or numeric type, '
                            . gettype($feature) . ' found.');
                    }
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
     * @return array
     */
    public function column(int $index) : array
    {
        return array_column($this->samples, $index);
    }

    /**
     * Return an array of column indices.
     *
     * @return array
     */
    public function indices() : array
    {
        return array_keys($this->samples[0] ?? []);
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
     * Return a tuple containing the size of the dataframe.
     *
     * @var array
     */
    public function size() : array
    {
        return [$this->numRows(), $this->numColumns()];
    }

    /**
     * Rotate the sample matrix and return it in an array. i.e. rows become
     * columns and columns become rows. This is equivalent to transposing.
     *
     * @return array
     */
    public function rotate() : array
    {
        if ($this->numRows() > 1) {
            $rotated = array_map(null, ...$this->samples);
        } else {
            $n = $this->numColumns();

            $rotated = [];

            for ($i = 0; $i < $n; $i++) {
                $rotated[$i] = array_column($this->samples, $i);
            }
        }

        return $rotated;
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
            throw new InvalidArgumentException('Sample not found at the given'
                . ' index ' . (string) $index . '.');
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
