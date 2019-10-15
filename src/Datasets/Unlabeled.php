<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;

use const Rubix\ML\PHI;

/**
 * Unlabeled
 *
 * Unlabeled datasets can be used to train *unsupervised* Estimators and for
 * feeding data into an Estimator to make predictions.
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
     * @param array $samples
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
     * Build a dataset from an iterator.
     *
     * @param iterable $samples
     * @return self
     */
    public static function fromIterator(iterable $samples) : self
    {
        $samples = is_array($samples)
            ? $samples
            : iterator_to_array($samples, false);

        return self::build($samples);
    }

    /**
     * Stack a number of datasets on top of each other to form a single
     * dataset.
     *
     * @param array $datasets
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function stack(array $datasets) : self
    {
        $samples = $labels = [];

        foreach ($datasets as $dataset) {
            if (!$dataset instanceof Dataset) {
                throw new InvalidArgumentException('Dataset must be an'
                    . ' instance of Dataset, ' . get_class($dataset)
                    . ' given.');
            }

            $samples = array_merge($samples, $dataset->samples());
        }

        return self::quick($samples);
    }

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param int $n
     * @throws \InvalidArgumentException
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
     * @throws \InvalidArgumentException
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
     * @throws \InvalidArgumentException
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
     * @throws \InvalidArgumentException
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
     * Prepend this dataset with another dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function prepend(Dataset $dataset) : Dataset
    {
        return self::quick(array_merge($dataset->samples(), $this->samples));
    }

    /**
     * Append this dataset with another dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function append(Dataset $dataset) : Dataset
    {
        return self::quick(array_merge($this->samples, $dataset->samples()));
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
     * Run a filter over the dataset using the values of a given column.
     *
     * @param int $index
     * @param callable $callback
     * @return self
     */
    public function filterByColumn(int $index, callable $callback) : self
    {
        $samples = [];

        foreach ($this->samples as $sample) {
            if ($callback($sample[$index])) {
                $samples[] = $sample;
            }
        }

        return self::quick($samples);
    }

    /**
     * Sort the dataset in place by a column in the sample matrix.
     *
     * @param int $index
     * @param bool $descending
     * @return self
     */
    public function sortByColumn(int $index, bool $descending = false) : self
    {
        $column = $this->column($index);

        array_multisort($column, $this->samples, $descending ? SORT_DESC : SORT_ASC);

        return $this;
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param float $ratio
     * @throws \InvalidArgumentException
     * @return array
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0. or $ratio >= 1.) {
            throw new InvalidArgumentException('Split ratio must be strictly'
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
     * @throws \InvalidArgumentException
     * @return array
     */
    public function fold(int $k = 3) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot create less than 2'
                . " folds, $k given.");
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
     * @return array
     */
    public function batch(int $n = 50) : array
    {
        return array_map([self::class, 'quick'], array_chunk($this->samples, $n));
    }

    /**
     * Partition the dataset into left and right subsets by a specified feature
     * column. The dataset is split such that, for categorical values, the left
     * subset contains all samples that match the value and the right side
     * contains samples that do not match. For continuous values, the left side
     * contains all the  samples that are less than the target value, and the
     * right side contains the samples that are greater than or equal to the
     * value.
     *
     * @param int $column
     * @param mixed $value
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function partition(int $column, $value) : array
    {
        if (!is_string($value) and !is_numeric($value)) {
            throw new InvalidArgumentException('Value must be a string or'
                . ' numeric type, ' . gettype($value) . ' given.');
        }

        $left = $right = [];

        if ($this->columnType($column) === DataType::CATEGORICAL) {
            foreach ($this->samples as $sample) {
                if ($sample[$column] === $value) {
                    $left[] = $sample;
                } else {
                    $right[] = $sample;
                }
            }
        } else {
            foreach ($this->samples as $sample) {
                if ($sample[$column] < $value) {
                    $left[] = $sample;
                } else {
                    $right[] = $sample;
                }
            }
        }

        return [
            self::quick($left),
            self::quick($right),
        ];
    }

    /**
     * Partition the dataset into left and right subsets based on their distance
     * between two centroids.
     *
     * @param array $leftCentroid
     * @param array $rightCentroid
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @throws \InvalidArgumentException
     * @return array
     */
    public function spatialPartition(array $leftCentroid, array $rightCentroid, Distance $kernel)
    {
        if (count($leftCentroid) !== count($rightCentroid)) {
            throw new InvalidArgumentException('Dimensionality mismatch between'
                . ' left and right centroids.');
        }

        $left = $right = [];

        foreach ($this->samples as $sample) {
            $lHat = $kernel->compute($sample, $leftCentroid);
            $rHat = $kernel->compute($sample, $rightCentroid);

            if ($lHat < $rHat) {
                $left[] = $sample;
            } else {
                $right[] = $sample;
            }
        }

        return [
            self::quick($left),
            self::quick($right),
        ];
    }

    /**
     * Generate a random subset without replacement.
     *
     * @param int $n
     * @throws \InvalidArgumentException
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

        $indices = array_rand($this->samples, $n);

        $indices = is_array($indices) ? $indices : [$indices];
        
        $samples = [];

        foreach ($indices as $index) {
            $samples[] = $this->samples[$index];
        }

        return self::quick($samples);
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param int $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubsetWithReplacement(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . " less than 1 sample, $n given.");
        }

        $maxIndex = $this->numRows() - 1;

        $subset = [];

        while (count($subset) < $n) {
            $subset[] = $this->samples[rand(0, $maxIndex)];
        }

        return self::quick($subset);
    }

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param int $n
     * @param array $weights
     * @throws \InvalidArgumentException
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

        $total = array_sum($weights);
        $max = (int) round($total * PHI);

        $subset = [];

        while (count($subset) < $n) {
            $delta = rand(0, $max) / PHI;

            foreach ($weights as $index => $weight) {
                $delta -= $weight;

                if ($delta <= 0.) {
                    $subset[] = $this->samples[$index];
                    
                    break 1;
                }
            }
        }

        return self::quick($subset);
    }

    /**
     * Return a dataset with all duplicate rows removed.
     *
     * @return self
     */
    public function deduplicate() : self
    {
        return self::quick(array_unique($this->samples, SORT_REGULAR));
    }
}
