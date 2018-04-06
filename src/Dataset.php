<?php

namespace Rubix\Engine;

use Rubix\Engine\Preprocessors\Preprocessor;
use InvalidArgumentException;
use Countable;

class Dataset implements Countable
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
     * The type of each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $types = [
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

        $this->types = array_map(function ($feature) {
            return is_string($feature) ? self::CATEGORICAL : self::CONTINUOUS;
        }, $samples[0] ?? []);

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
        return count($this->types);
    }

    /**
     * @return array
     */
    public function types() : array
    {
        return $this->types;
    }

    /**
     * Have a preprocessor transform the dataset.
     *
     * @param  \Rubix\Engine\Preprocessors\Preprocessor  $preprocessor
     * @return void
     */
    public function transform(Preprocessor $preprocessor) : void
    {
        $preprocessor->transform($this->samples);
    }

    /**
     * Randomize the dataset.
     *
     * @return self
     */
    public function randomize()
    {
        shuffle($this->samples);

        return $this;
    }

    /**
     * Take n samples and outcomes from this dataset and return them in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function take(int $n = 1)
    {
        return new self(array_splice($this->samples, 0, $n));
    }

    /**
     * Leave n samples and outcomes on this dataset and return the rest in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function leave(int $n = 1)
    {
        return new self(array_splice($this->samples, $n));
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return array
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0.0 || $ratio >= 1.0) {
            throw new InvalidArgumentException('Split ratio must be a float value between 0 and 1.');
        }

        $testing = array_splice($this->samples, ceil($ratio * $this->rows()));

        return [
            new self($this->samples),
            new self($testing),
        ];
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomSubset(float $ratio = 0.1)
    {
        if ($ratio <= 0.0 || $ratio >= 1.0) {
            throw new InvalidArgumentException('Split ratio must be a float value between 0 and 1.');
        }

        $subset = [];

        foreach (range(1, ceil($ratio * $this->rows())) as $i) {
            $subset[] = $this->samples[array_rand($this->samples)];
        }

        return new self($subset);
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
     * @return array
     */
    public function toArray() : array
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
}
