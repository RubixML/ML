<?php

namespace Rubix\Engine\Datasets;

use Rubix\Engine\Transformers\Transformer;
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
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $samples)
    {
        foreach ($samples as &$sample) {
            $sample = array_values((array) $sample);

            if (count($sample) !== count(current($samples))) {
                throw new InvalidArgumentException('The number of feature columns'
                 . ' must be equal for all samples.');
            }

            foreach ($sample as &$feature) {
                if (!is_string($feature) and !is_numeric($feature)) {
                    throw new InvalidArgumentException('Feature must be a string'
                    . ' or numeric, ' . gettype($feature) . ' found.');
                }
            }
        }

        $this->samples = array_values($samples);
    }

    /**
     * @return array
     */
    public function samples() : array
    {
        return $this->samples;
    }

    /**
     * Return the sample at the given row index.
     *
     * @param  mixed  $index
     * @return array
     */
    public function row($index) : array
    {
        if (!isset($this->samples[$index])) {
            throw new RuntimeException('Sample not found at the given index '
                . (string) $index . '.');
        }

        return $this->samples[$index];
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
     * @param  mixed  $index
     * @return array
     */
    public function column($index) : array
    {
        if (!$this->offsetExists($index)) {
            throw new RuntimeException('Feature column not found at the given'
            . ' index ' . (string) $index . '.');
        }

        return array_column($this->samples, $index);
    }

    /**
     * Return an array of autodetected column types.
     *
     * @return array
     */
    public function columnTypes() : array
    {
        return array_map(function ($feature) {
            return is_string($feature) ? self::CATEGORICAL : self::CONTINUOUS;
        }, $this->samples[0] ?? []);
    }

    /**
     * Return the number of feature columns in the datasets.
     *
     * @return int
     */
    public function numColumns() : int
    {
        return count($this->samples[0] ?? []);
    }

    /**
     * Have a transformer transform the dataset.
     *
     * @param  \Rubix\Engine\Transformers\Transformer  $transformer
     * @return void
     */
    public function transform(Transformer $transformer) : void
    {
        $transformer->transform($this->samples);
    }

    /**
     * Returns an array of feature columns.
     *
     * @return array
     */
    public function rotate() : array
    {
        return array_map(null, ...$this->samples);
    }

    /**
     * @return int
     */
    public function count() : int
    {
        return $this->numRows();
    }

    /**
     * Is the dataset empty?
     *
     * @return bool
     */
    public function isEmpty() : bool
    {
        return $this->numRows() === 0;
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
        return isset($this->samples[0][$index]);
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
     * Return a column from the dataset given by index or if an array is passed
     * return an array of columns by their index.
     *
     * @param  mixed  $indices
     * @return array
     */
    public function offsetGet($indices) : array
    {
        if (is_array($indices)) {
            $columns = [];

            foreach ($indices as $index) {
                $columns[] = $this->column($index);
            }

            return $columns;
        }

        return $this->column($indices);
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
