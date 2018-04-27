<?php

namespace Rubix\Engine\Datasets;

use InvalidArgumentException;

class Unsupervised extends Dataset
{
    /**
     * Build a dataset from an iterator.
     *
     * @param  iterable  $data
     * @return self
     */
    public static function fromIterator(iterable $data) : self
    {
        return new self(is_array($data) ? $data : iterator_to_array($data));
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
     * Return a dataset containing only the first n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function head(int $n = 10) : self
    {
        return new self(array_slice($this->samples, 0, $n));
    }

    /**
     * Take n samples from this dataset and return them in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function take(int $n = 1) : self
    {
        return new self(array_splice($this->samples, 0, $n));
    }

    /**
     * Leave n samples on this dataset and return the rest in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function leave(int $n = 1) : self
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
            throw new InvalidArgumentException('Sample ratio must be a float value between 0 and 1.');
        }

        $left = array_splice($this->samples, round($ratio * $this->rows()));

        return [
            new self($left),
            new self($this->samples),
        ];
    }

    /**
     * Divide the dataset into n sets of equal proportion.
     *
     * @param  int  $n
     * @return array
     */
    public function divide(int $n = 5) : array
    {
        $size = round($this->rows() / $sets);

        $subsets = [];

        while (!empty($this->samples)) {
            $subsets[] = new self(array_splice($this->samples, 0, $size));
        }

        return $subsets;
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomSubset(float $ratio = 0.1) : self
    {
        if ($ratio <= 0.0 || $ratio >= 1.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value between 0 and 1.');
        }

        $subset = $this->samples;

        shuffle($subset);

        return new self(array_slice($subset, 0, round($ratio * $this->rows())));
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomSubsetWithReplacement(float $ratio = 0.1) : self
    {
        if ($ratio <= 0.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value greater than 0.');
        }

        $subset = [];

        foreach (range(1, round($ratio * $this->rows())) as $i) {
            $subset[] = $this->samples[array_rand($this->samples)];
        }

        return new self($subset);
    }
}
