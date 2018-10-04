<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Transformers\Transformer;
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
                        . ' columns must be equal for all samples.');
                }

                foreach ($sample as $feature) {
                    if (!is_string($feature) and !is_numeric($feature)) {
                        throw new InvalidArgumentException('Feature must be a'
                            . ' string or numeric type, '
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
     * Return an array representing the indices of each feature column.
     *
     * @return array
     */
    public function axes() : array
    {
        return array_keys($this->samples[0] ?? []);
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
        if (!isset($this->samples[0])) {
            return [];
        }

        return array_map(function ($feature) {
            return is_string($feature) ? self::CATEGORICAL : self::CONTINUOUS;
        }, $this->samples[0]);
    }

    /**
     * Get the datatype for a feature column given a column index.
     *
     * @param  int  $index
     * @return int
     */
    public function columnType(int $index) : int
    {
        return is_string($this->samples[0][$index])
            ? self::CATEGORICAL
            : self::CONTINUOUS;
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
     * Apply a transformation to the dataframe.
     *
     * @param  \Rubix\ML\Transformers\Transformer  $transformer
     * @return void
     */
    public function apply(Transformer $transformer) : void
    {
        $transformer->transform($this->samples);
    }

    /**
     * Rotate the dataframe and return it in an array. i.e. rows become
     * columns and columns become rows. This operation is equivalent to
     * transposing a matrix.
     *
     * @return self
     */
    public function rotate() : self
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

        return new self($rotated);
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
