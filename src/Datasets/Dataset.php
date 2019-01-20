<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Transformers\Transformer;
use IteratorAggregate;
use JsonSerializable;
use ArrayAccess;
use Countable;

interface Dataset extends ArrayAccess, IteratorAggregate, JsonSerializable, Countable
{
    const PHI = 100000000;

    /**
     * Stack a number of datasets on top of each other to form a single
     * dataset.
     * 
     * @param  array  $datasets
     * @return self
     */
    public static function stack(array $datasets);

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
     * Return an array of feature column datatypes autodectected using the first
     * sample in the dataframe.
     *
     * @return array
     */
    public function types() : array;

    /**
     * Return the number of columns match a specific data type.
     * 
     * @param  int  $type
     * @return int
     */
    public function typeCount(int $type) : int;

        /**
     * Return the number of feature columns for each data type.
     * 
     * @return int[]
     */
    public function typeCounts() : array;

    /**
     * Get the datatype for a feature column given a column index.
     *
     * @param  int  $index
     * @return int|null
     */
    public function columnType(int $index) : ?int;

    /**
     * Return the number of feature columns in the datasets.
     *
     * @return int
     */
    public function numColumns() : int;

    /**
     * Return a tuple containing the shape of the dataframe i.e the number of
     * rows and columns.
     *
     * @var array
     */
    public function shape() : array;

    /**
     * Return the number of elements in the dataframe.
     *
     * @return int
     */
    public function size() : int;

    /**
     * Apply a tranformation to the sample matrix.
     *
     * @param  \Rubix\ML\Transformers\Transformer  $transformer
     * @return self
     */
    public function apply(Transformer $transformer);

    /**
     * Rotate the dataframe and return it in an array. i.e. rows become
     * columns and columns become rows.
     *
     * @return array
     */
    public function columns() : array;

    /**
     * Return the columns that match a given data type.
     * 
     * @param  int  $type
     * @return array
     */
    public function columnsByType(int $type) : array;

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
     * Run a filter over the dataset using the values of a given column.
     *
     * @param  int  $index
     * @param  callable  $fn
     * @return self
     */
    public function filterByColumn(int $index, callable $fn);

    /**
     * Sort the dataset by a column in the sample matrix.
     *
     * @param  int  $index
     * @param  bool  $descending
     * @return self
     */
    public function sortByColumn(int $index, bool $descending = false);

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
     * Partition the dataset into left and right subsets by a specified feature
     * column.
     *
     * @param  int  $index
     * @param  mixed  $value
     * @return array
     */
    public function partition(int $index, $value) : array;

    /**
     * Generate a random subset of n samples with replacement.
     *
     * @param  int  $n
     * @return self
     */
    public function randomSubsetWithReplacement(int $n);

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param  int  $n
     * @param  array  $weights
     * @return self
     */
    public function randomWeightedSubsetWithReplacement(int $n, array $weights);

    /**
     * Merge this dataset with another dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function merge(Dataset $dataset) : Dataset;

    /**
     * Is the dataset empty?
     *
     * @return bool
     */
    public function empty() : bool;
}
