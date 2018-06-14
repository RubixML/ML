<?php

namespace Rubix\ML\Datasets;

use IteratorAggregate;
use ArrayAccess;
use Countable;

interface Dataset extends ArrayAccess, IteratorAggregate, Countable
{
    /**
     * Factory method to create a dataset from an array of datasets.
     *
     * @param  array  $datasets
     * @return self
     */
    public static function combine(array $datasets);

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function head(int $n = 10);

    /**
     * Return a dataset containing only the last n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function tail(int $n = 10);

    /**
     * Take n samples from the dataset and return them in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function take(int $n = 1);

    /**
     * Leave n samples on the dataset and return the rest in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function leave(int $n = 1);

    /**
     * Randomize the dataset.
     *
     * @return self
     */
    public function randomize();

    /**
     * Split the dataset into two subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @return array
     */
    public function split(float $ratio = 0.5) : array;

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param  int  $k
     * @return array
     */
    public function fold(int $k = 10) : array;

    /**
     * Generate a collection of batches of size n from the dataset. If there are
     * not enough samples to fill an entire batch, then the dataset will contain
     * as many samples as possible.
     *
     * @param  int  $n
     * @return array
     */
    public function batch(int $n = 50) : array;

    /**
     * Generate a random subset of n samples with replacement.
     *
     * @param  int  $n
     * @return self
     */
    public function randomSubset(int $n = 1);

    /**
     * Return the 2-dimensional sample matrix.
     *
     * @return array
     */
    public function samples() : array;

    /**
     * Return an array with all the data.
     *
     * @return array
     */
    public function all() : array;
}
