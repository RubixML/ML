<?php

namespace Rubix\ML\Datasets;

use InvalidArgumentException;

class Unlabeled extends DataFrame implements Dataset
{
    /**
     * Factory method to create an unsupervised dataset from an array of datasets.
     *
     * @param  array  $datasets
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function combine(array $datasets = []) : self
    {
        $samples = [];

        foreach ($datasets as $dataset) {
            if (!$dataset instanceof Dataset) {
                throw new InvalidArgumentException('Cannot merge any non'
                    . ' datasets, ' . get_class($dataset) . ' found.');
            }

            $samples = array_merge($samples, $dataset->samples());
        }

        return new self($samples);
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
    public function tail(int $n = 10) : self
    {
        return new self(array_slice($this->samples, -$n));
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
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return array
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0 or $ratio >= 1) {
            throw new InvalidArgumentException('Split ratio must be strictly'
            . ' between 0 and 1.');
        }

        $n = round($ratio * $this->numRows());

        return [
            new self(array_slice($this->samples, 0, $n)),
            new self(array_slice($this->samples, $n)),
        ];
    }

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return array
     */
    public function fold(int $k = 3) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot fold the dataset less'
                . ' than 1 time.');
        }

        $samples = $this->samples;

        $n = round(count($samples) / $k);

        $folds = [];

        for ($i = 0; $i < $k; $i++) {
            $folds[] = new self(array_splice($samples, 0, $n));
        }

        return $folds;
    }

    /**
     * Generate a collection of batches of size n from the dataset. If there are
     * not enough samples to fill an entire batch, then the dataset will contain
     * as many samples as possible.
     *
     * @param  int  $n
     * @return array
     */
    public function batch(int $n = 50) : array
    {
        $batches = [];

        $samples = $this->samples;

        while (!empty($samples)) {
            $batches[] = new self(array_splice($samples, 0, $n));
        }

        return $batches;
    }

    /**
     * Generate a random subset.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubset(int $n = 1) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . ' less than 1 sample.');
        }

        if ($n > $this->numRows()) {
            throw new InvalidArgumentException('Cannot generate a larger subset'
                . ' than the sample size.');
        }

        return new self(array_intersect_key($this->samples,
            array_rand($this->samples, $n)));
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubsetWithReplacement(int $n = 1) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . ' less than 1 sample.');
        }

        $subset = [];

        for ($i = 0; $i < $n; $i++) {
            $subset[] = $this->samples[array_rand($this->samples)];
        }

        return new self($subset);
    }
}
