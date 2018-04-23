<?php

namespace Rubix\Engine;

use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Persisters\Persistable;
use InvalidArgumentException;
use IteratorAggregate;
use ArrayIterator;
use Countable;

class Dataset implements Persistable, IteratorAggregate, Countable
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    /**
     * The feature vectors or columns of a data table.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * @param  array  $samples
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $samples)
    {
        foreach ($samples as $i => &$sample) {
            if (!is_array($sample)) {
                $sample = [$sample];
            }

            if (count($sample) !== count($samples[0])) {
                throw new InvalidArgumentException('The number of feature columns must be equal for all samples.');
            }

            foreach ($sample as &$feature) {
                if (!is_string($feature) && !is_numeric($feature)) {
                    throw new InvalidArgumentException('Feature values must be a string or numeric type, ' . gettype($feature) . ' found.');
                }

                if (is_string($feature) && is_numeric($feature)) {
                    $feature = $this->convertNumericString($feature);
                }
            }
        }

        $this->samples = $samples;
    }

    /**
     * @return array
     */
    public function samples() : array
    {
        return $this->samples;
    }

    /**
     * Return a sample given by row number. Return null if not found.
     *
     * @param  int  $row
     * @return array|null
     */
    public function getSample(int $row) : ?array
    {
        return $this->samples[$row] ?? null;
    }

    /**
     * @return int
     */
    public function rows() : int
    {
        return count($this->samples);
    }

    /**
     * The number of feature columns in this dataset.
     *
     * @return int
     */
    public function columns() : int
    {
        return count($this->samples[0] ?? []);
    }

    /**
     * @return array
     */
    public function columnTypes() : array
    {
        return array_map(function ($feature) {
            return is_string($feature) ? self::CATEGORICAL : self::CONTINUOUS;
        }, $this->samples[0] ?? []);
    }

    /**
     * Have a transformer transform the dataset.
     *
     * @param  \Rubix\Engine\Contracts\Transformer  $transformer
     * @return void
     */
    public function transform(Transformer $transformer) : void
    {
        $transformer->transform($this->samples);
    }

    /**
     * Rotates the table of samples into columns of feature values.
     *
     * @return array
     */
    public function rotate() : array
    {
        return array_map(null, ...$this->samples);
    }

    /**
     * Remove a feature column from the dataset given by the column's offset.
     *
     * @param  int  $offset
     * @return self
     */
    public function removeColumn(int $offset) : self
    {
        foreach ($this->samples as &$sample) {
            unset($sample[$offset]);

            $sample = array_values($sample);
        }

        unset($this->types[$offset]);

        $this->types = array_values($this->types);
    }

    /**
     * Convert a numeric string into its appropriate data type.
     *
     * @param  string  $string
     * @return mixed
     */
    public function convertNumericString(string $string)
    {
        return is_float($string + 0) ? (float) $string : (int) $string;
    }

    /**
     * Return an array containing all of the samples.
     *
     * @return array
     */
    public function all() : array
    {
        return $this->samples;
    }

    /**
     * @return int
     */
    public function count() : int
    {
        return $this->rows();
    }

    /**
     * Is the dataset empty?
     *
     * @return bool
     */
    public function isEmpty() : bool
    {
        return $this->rows() === 0;
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

    /**
     * @param  int  $row
     * @return mixed
     */
    public function __get(int $row)
    {
        return $this->getSample($row);
    }

    /**
     * @param  int  $row
     * @return bool
     */
    public function __isset(int $row)
    {
        return isset($this->samples[$row]);
    }
}
