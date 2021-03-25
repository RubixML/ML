<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Other\Helpers\Console;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Generator;

use function Rubix\ML\warn_deprecated;
use function count;
use function array_slice;

use const Rubix\ML\PHI;

/**
 * Unlabeled
 *
 * Unlabeled datasets are used to train unsupervised learners and for feeding unknown
 * samples into an estimator to make predictions during inference. As their name implies,
 * they do not require a corresponding label for each sample.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Unlabeled extends Dataset
{
    /**
     * Build a new unlabeled dataset with validation.
     *
     * @param array[] $samples
     * @return self
     */
    public static function build(array $samples = []) : self
    {
        return new self($samples, true);
    }

    /**
     * Build a new unlabeled dataset foregoing validation.
     *
     * @param array[] $samples
     * @return self
     */
    public static function quick(array $samples = []) : self
    {
        return new self($samples, false);
    }

    /**
     * Build a dataset with the rows from an iterable data table.
     *
     * @param iterable<array> $iterator
     * @return self
     */
    public static function fromIterator(iterable $iterator) : self
    {
        $samples = is_array($iterator) ? $iterator : iterator_to_array($iterator, false);

        return self::build($samples);
    }

    /**
     * Stack a number of datasets on top of each other to form a single
     * dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset[] $datasets
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public static function stack(array $datasets) : self
    {
        $n = $datasets[array_key_first($datasets)]->numColumns();

        $samples = [];

        foreach ($datasets as $dataset) {
            if (!$dataset instanceof Dataset) {
                throw new InvalidArgumentException('Dataset must implement'
                    . ' the Dataset interface.');
            }

            if ($dataset->numColumns() !== $n) {
                throw new InvalidArgumentException('Dataset must have'
                    . " the same number of columns, $n expected but"
                    . " {$dataset->numColumns()} given.");
            }

            $samples[] = $dataset->samples();
        }

        return self::quick(array_merge(...$samples));
    }

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function head(int $n = 10) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('The number of samples'
                . " cannot be less than 1, $n given.");
        }

        return $this->slice(0, $n);
    }

    /**
     * Return a dataset containing only the last n samples.
     *
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function tail(int $n = 10) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('The number of samples'
                . " cannot be less than 1, $n given.");
        }

        return $this->slice(-$n, $this->numRows());
    }

    /**
     * Take n samples from this dataset and return them in a new dataset.
     *
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function take(int $n = 1) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('The number of samples'
                . " cannot be less than 1, $n given.");
        }

        return $this->splice(0, $n);
    }

    /**
     * Leave n samples on this dataset and return the rest in a new dataset.
     *
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function leave(int $n = 1) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('The number of samples'
                . " cannot be less than 1, $n given.");
        }

        return $this->splice($n, $this->numRows());
    }

    /**
     * Return an n size portion of the dataset in a new dataset.
     *
     * @param int $offset
     * @param int $n
     * @return self
     */
    public function slice(int $offset, int $n) : self
    {
        return self::quick(array_slice($this->samples, $offset, $n));
    }

    /**
     * Remove a size n chunk of the dataset starting at offset and return it in
     * a new dataset.
     *
     * @param int $offset
     * @param int $n
     * @return self
     */
    public function splice(int $offset, int $n) : self
    {
        return self::quick(array_splice($this->samples, $offset, $n));
    }

    /**
     * Merge another dataset with this dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function merge(Dataset $dataset) : self
    {
        if (!$dataset->empty() and !$this->empty()) {
            if ($dataset->numColumns() !== $this->numColumns()) {
                throw new InvalidArgumentException('Datasets must have'
                    . " the same dimensionality, {$this->numColumns()}"
                    . " expected, but {$dataset->numColumns()} given.");
            }
        }

        return self::quick(array_merge($this->samples, $dataset->samples()));
    }

    /**
     * Join the columns of this dataset with another dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function join(Dataset $dataset) : self
    {
        if ($dataset->numRows() !== $this->numRows()) {
            throw new InvalidArgumentException('Datasets must have'
                . " the same number of rows, {$this->numRows()}"
                . " expected, but {$dataset->numRows()} given.");
        }

        $samples = [];

        foreach ($this->samples as $i => $sample) {
            $samples[] = array_merge($sample, $dataset->sample($i));
        }

        return self::quick($samples);
    }

    /**
     * Merge the columns of this dataset with another dataset.
     *
     * @deprecated
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return self
     */
    public function augment(Dataset $dataset) : self
    {
        warn_deprecated('Augment() is deprecated, use join() instead.');

        return $this->join($dataset);
    }

    /**
     * Drop the row at the given offset.
     *
     * @param int $offset
     * @return self
     */
    public function dropRow(int $offset) : self
    {
        return $this->dropRows([$offset]);
    }

    /**
     * Drop the rows at the given indices.
     *
     * @param int[] $offsets
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function dropRows(array $offsets) : self
    {
        foreach ($offsets as $offset) {
            unset($this->samples[$offset]);
        }

        $this->samples = array_values($this->samples);

        return $this;
    }

    /**
     * Randomize the dataset in place and return self for chaining.
     *
     * @return self
     */
    public function randomize() : self
    {
        shuffle($this->samples);

        return $this;
    }

    /**
     * Filter the rows of the dataset using the values of a feature column as the
     * argument to a callback.
     *
     * @param int $offset
     * @param callable $callback
     * @return self
     */
    public function filterByColumn(int $offset, callable $callback) : self
    {
        $samples = [];

        foreach ($this->samples as $sample) {
            if ($callback($sample[$offset])) {
                $samples[] = $sample;
            }
        }

        return self::quick($samples);
    }

    /**
     * Sort the dataset in place by a column in the sample matrix.
     *
     * @param int $offset
     * @param bool $descending
     * @return self
     */
    public function sortByColumn(int $offset, bool $descending = false) : self
    {
        $column = $this->column($offset);

        array_multisort($column, $this->samples, $descending ? SORT_DESC : SORT_ASC);

        return $this;
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param float $ratio
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return array{self,self}
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio < 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        $n = (int) floor($ratio * $this->numRows());

        return [
            self::quick(array_slice($this->samples, 0, $n)),
            self::quick(array_slice($this->samples, $n)),
        ];
    }

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param int $k
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return list<self>
     */
    public function fold(int $k = 3) : array
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Cannot create less than 1'
                . " fold, $k given.");
        }

        $samples = $this->samples;

        $n = (int) floor(count($samples) / $k);

        $folds = [];

        while (count($folds) < $k) {
            $folds[] = self::quick(array_splice($samples, 0, $n));
        }

        return $folds;
    }

    /**
     * Generate a collection of batches of size n from the dataset. If there are
     * not enough samples to fill an entire batch, then the dataset will contain
     * as many samples as possible.
     *
     * @param int $n
     * @return list<self>
     */
    public function batch(int $n = 50) : array
    {
        return array_map([self::class, 'quick'], array_chunk($this->samples, $n));
    }

    /**
     * Partition the dataset into left and right subsets using the values of a single feature column for comparison.
     *
     * @internal
     *
     * @param int $column
     * @param string|int|float $value
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return array{self,self}
     */
    public function splitByColumn(int $column, $value) : array
    {
        $left = $right = [];

        if ($this->columnType($column)->isContinuous()) {
            foreach ($this->samples as $sample) {
                if ($sample[$column] <= $value) {
                    $left[] = $sample;
                } else {
                    $right[] = $sample;
                }
            }
        } else {
            foreach ($this->samples as $sample) {
                if ($sample[$column] === $value) {
                    $left[] = $sample;
                } else {
                    $right[] = $sample;
                }
            }
        }

        return [self::quick($left), self::quick($right)];
    }

    /**
     * Partition the dataset into left and right subsets based on the samples' distances from two centroids.
     *
     * @internal
     *
     * @param (string|int|float)[] $leftCentroid
     * @param (string|int|float)[] $rightCentroid
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return array{self,self}
     */
    public function spatialSplit(array $leftCentroid, array $rightCentroid, Distance $kernel)
    {
        $left = $right = [];

        foreach ($this->samples as $sample) {
            $lDistance = $kernel->compute($sample, $leftCentroid);
            $rDistance = $kernel->compute($sample, $rightCentroid);

            if ($lDistance < $rDistance) {
                $left[] = $sample;
            } else {
                $right[] = $sample;
            }
        }

        return [self::quick($left), self::quick($right)];
    }

    /**
     * Generate a random subset without replacement.
     *
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function randomSubset(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a'
                . " subset of less than 1 sample, $n given.");
        }

        if ($n > $this->numRows()) {
            throw new InvalidArgumentException('Cannot generate subset'
                . " of more than {$this->numRows()}, $n given.");
        }

        $offsets = array_rand($this->samples, $n);

        $offsets = is_array($offsets) ? $offsets : [$offsets];

        $samples = [];

        foreach ($offsets as $offset) {
            $samples[] = $this->samples[$offset];
        }

        return self::quick($samples);
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function randomSubsetWithReplacement(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . " less than 1 sample, $n given.");
        }

        $maxOffset = $this->numRows() - 1;

        $samples = [];

        while (count($samples) < $n) {
            $samples[] = $this->samples[rand(0, $maxOffset)];
        }

        return self::quick($samples);
    }

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param int $n
     * @param (int|float)[] $weights
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return self
     */
    public function randomWeightedSubsetWithReplacement(int $n, array $weights) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a'
                . " subset of less than 1 sample, $n given.");
        }

        if (count($weights) !== count($this->samples)) {
            throw new InvalidArgumentException('The number of weights'
                . ' must be equal to the number of samples in the'
                . ' dataset, ' . count($this->samples) . ' needed'
                . ' but ' . count($weights) . ' given.');
        }

        $numLevels = (int) round(sqrt(count($weights)));

        $levels = array_chunk($weights, $numLevels, true);
        $levelTotals = array_map('array_sum', $levels);

        $total = array_sum($levelTotals);
        $max = (int) round($total * PHI);

        $samples = [];

        while (count($samples) < $n) {
            $delta = rand(0, $max) / PHI;

            foreach ($levels as $i => $level) {
                $levelTotal = $levelTotals[$i];

                if ($delta - $levelTotal > 0) {
                    $delta -= $levelTotal;

                    continue;
                }

                foreach ($level as $offset => $weight) {
                    $delta -= $weight;

                    if ($delta <= 0.0) {
                        $samples[] = $this->samples[$offset];

                        break;
                    }
                }
            }
        }

        return self::quick($samples);
    }

    /**
     * Remove duplicate rows from the dataset.
     *
     * @return self
     */
    public function deduplicate() : self
    {
        $this->samples = array_values(array_unique($this->samples, SORT_REGULAR));

        return $this;
    }

    /**
     * Return the dataset object as a data table array.
     *
     * @return array[]
     */
    public function toArray() : array
    {
        return $this->samples;
    }

    /**
     * Return a row from the dataset at the given offset.
     *
     * @param int $offset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return array[]
     */
    public function offsetGet($offset) : array
    {
        if (isset($this->samples[$offset])) {
            return $this->samples[$offset];
        }

        throw new InvalidArgumentException("Row at offset $offset not found.");
    }

    /**
     * Get an iterator for the samples in the dataset.
     *
     * @return \Generator<array>
     */
    public function getIterator() : Generator
    {
        yield from $this->samples;
    }

    /**
     * Return a string representation of the first few rows of the dataset.
     *
     * @return string
     */
    public function __toString() : string
    {
        [$tRows, $tCols] = Console::size();

        $m = (int) floor($tRows / 2) + 2;
        $n = (int) floor($tCols / (3 + Console::TABLE_CELL_WIDTH));

        $m = min($this->numRows(), $m);
        $n = min($this->numColumns(), $n);

        $header = [];

        for ($column = 0; $column < $n; ++$column) {
            $header[] = "Column $column";
        }

        $table = array_slice($this->samples, 0, $m);

        foreach ($table as $i => &$row) {
            $row = array_slice($row, 0, $n);
        }

        array_unshift($table, $header);
        $columnWidth = (int) floor($tCols) / count($table[0]);

        return Console::table($table, $columnWidth);
    }
}
