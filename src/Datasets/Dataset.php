<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Transformers\Transformer;
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
     * Return the 2-dimensional sample matrix.
     *
     * @return array
     */
    public function samples() : array;

    /**
     * Return the sample at the given row index.
     *
     * @param  int  $index
     * @return array
     */
    public function row(int $index) : array;

    /**
     * Return the number of rows in the datasets.
     *
     * @return int
     */
    public function numRows() : int;

    /**
     * Return the feature column at the given index.
     *
     * @param  int  $index
     * @return array
     */
    public function column(int $index) : array;

    /**
     * Return an array of column indices.
     *
     * @return array
     */
    public function indices() : array;

    /**
     * Return an array of autodetected feature column types.
     *
     * @return array
     */
    public function columnTypes() : array;

    /**
     * Get the column type for a given column index.
     *
     * @param  int  $index
     * @return int
     */
    public function type(int $index) : int;

    /**
     * Return the number of feature columns in the datasets.
     *
     * @return int
     */
    public function numColumns() : int;

    /**
     * Have a transformer transform the dataset.
     *
     * @param  \Rubix\ML\Transformers\Transformer  $transformer
     * @return void
     */
    public function transform(Transformer $transformer) : void;

    /**
     * Rotate the sample matrix.
     *
     * @return array
     */
    public function rotate() : array;

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
     * Remove a size n chunk of the dataset starting at offset and return it in
     * a new dataset.
     *
     * @param  int  $offset
     * @param  int  $n
     * @return self
     */
    public function splice(int $offset, int $n);

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
     * Generate a random subset of n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function randomSubset(int $n = 1);

    /**
     * Generate a random subset of n samples with replacement.
     *
     * @param  int  $n
     * @return self
     */
    public function randomSubsetWithReplacement(int $n = 1);
}
