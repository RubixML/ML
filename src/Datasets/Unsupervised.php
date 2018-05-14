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
    public function randomize() : self
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
     * Return a dataset containing only the last n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function tail(int $n = 10) : Dataset
    {
        return new self(array_slice($this->samples, -$n));
    }

    /**
     * Take n samples from this dataset and return them in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function take(int $n = 1) : Dataset
    {
        return new self(array_splice($this->samples, 0, $n));
    }

    /**
     * Leave n samples on this dataset and return the rest in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function leave(int $n = 1) : Dataset
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

        return [
            new self(array_splice($this->samples, round($ratio * $this->rows()))),
            new self($this->samples),
        ];
    }

    /**
     * Fold the dataset k times to form k + 1 equal size datasets.
     *
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return array
     */
    public function fold(int $k = 2) : array
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Cannot fold the dataset less than 1 time.');
        }

        $n = round(count($this->samples) / ($k + 1));
        $datasets = [];

        for ($i = 0; $i < $k + 1; $i++) {
            $datasets[] = new self(array_splice($this->samples, 0, $n));
        }

        return $datasets;
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomSubset(float $ratio = 0.1) : Dataset
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
    public function generateRandomSubsetWithReplacement(float $ratio = 0.1) : Dataset
    {
        if ($ratio <= 0.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value greater than 0.');
        }

        $subset = [];

        for ($i = 0; $i < round($ratio * $this->rows()); $i++) {
            $subset[] = $this->samples[array_rand($this->samples)];
        }

        return new self($subset);
    }
}
