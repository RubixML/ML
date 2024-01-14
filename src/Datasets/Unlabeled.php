<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Traversable;

use function count;
use function array_slice;
use function array_sum;
use function array_map;
use function array_chunk;
use function array_rand;
use function round;
use function sqrt;
use function getrandmax;
use function rand;

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
     * @param array<mixed[]> $samples
     * @return self
     */
    public static function build(array $samples = []) : self
    {
        return new self($samples, true);
    }

    /**
     * Build a new unlabeled dataset foregoing validation.
     *
     * @param array<mixed[]> $samples
     * @return self
     */
    public static function quick(array $samples = []) : self
    {
        return new self($samples, false);
    }

    /**
     * Build a dataset with the rows from an iterable data table.
     *
     * @param iterable<mixed[]> $iterator
     * @return self
     */
    public static function fromIterator(iterable $iterator) : self
    {
        $samples = is_array($iterator) ? $iterator : iterator_to_array($iterator, false);

        return self::build($samples);
    }

    /**
     * Stack a number of datasets on top of each other to form a single dataset.
     *
     * @param iterable<\Rubix\ML\Datasets\Dataset> $datasets
     * @throws InvalidArgumentException
     * @return self
     */
    public static function stack(iterable $datasets) : self
    {
        $samples = [];

        foreach ($datasets as $i => $dataset) {
            if (!$dataset instanceof Dataset) {
                throw new InvalidArgumentException('Dataset must implement'
                    . ' the Dataset interface.');
            }

            if ($dataset->empty()) {
                continue;
            }

            if (isset($lastNumFeatures) and $dataset->numFeatures() !== $lastNumFeatures) {
                throw new InvalidArgumentException("Dataset $i must have"
                    . " the same number of features, $lastNumFeatures"
                    . " expected but {$dataset->numFeatures()} given.");
            }

            $samples[] = $dataset->samples();

            $lastNumFeatures = $dataset->numFeatures();
        }

        return self::quick(array_merge(...$samples));
    }

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param int $n
     * @throws InvalidArgumentException
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
     * @throws InvalidArgumentException
     * @return self
     */
    public function tail(int $n = 10) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('The number of samples'
                . " cannot be less than 1, $n given.");
        }

        return $this->slice(-$n, $this->numSamples());
    }

    /**
     * Take n samples from this dataset and return them in a new dataset.
     *
     * @param int $n
     * @throws InvalidArgumentException
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
     * @throws InvalidArgumentException
     * @return self
     */
    public function leave(int $n = 1) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('The number of samples'
                . " cannot be less than 1, $n given.");
        }

        return $this->splice($n, $this->numSamples());
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
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     * @return self
     */
    public function merge(Dataset $dataset) : self
    {
        if (!$dataset->empty() and !$this->empty()) {
            if ($dataset->numFeatures() !== $this->numFeatures()) {
                throw new InvalidArgumentException('Datasets must have'
                    . " the same dimensionality, {$this->numFeatures()}"
                    . " expected, but {$dataset->numFeatures()} given.");
            }
        }

        return self::quick(array_merge($this->samples, $dataset->samples()));
    }

    /**
     * Join the columns of this dataset with another dataset.
     *
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     * @return self
     */
    public function join(Dataset $dataset) : self
    {
        if ($dataset->numSamples() !== $this->numSamples()) {
            throw new InvalidArgumentException('Datasets must have'
                . " the same number of rows, {$this->numSamples()}"
                . " expected, but {$dataset->numSamples()} given.");
        }

        $samples = [];

        foreach ($this->samples as $i => $sample) {
            $samples[] = array_merge($sample, $dataset->sample($i));
        }

        return self::quick($samples);
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
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param float $ratio
     * @throws InvalidArgumentException
     * @return array{self,self}
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio < 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        $n = (int) floor($ratio * $this->numSamples());

        $left = self::quick(array_slice($this->samples, 0, $n));
        $right = self::quick(array_slice($this->samples, $n));

        return [$left, $right];
    }

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param int $k
     * @throws InvalidArgumentException
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
     * @param positive-int $n
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
     * @throws InvalidArgumentException
     * @return array{self,self}
     */
    public function splitByFeature(int $column, $value) : array
    {
        $left = $right = [];

        if ($this->featureType($column)->isContinuous()) {
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
     * @param Distance $kernel
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
     * @throws InvalidArgumentException
     * @return self
     */
    public function randomSubset(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a'
                . " subset of less than 1 sample, $n given.");
        }

        if ($n > $this->numSamples()) {
            throw new InvalidArgumentException('Cannot generate subset'
                . " of more than {$this->numSamples()}, $n given.");
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
     * @throws InvalidArgumentException
     * @return self
     */
    public function randomSubsetWithReplacement(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . " less than 1 sample, $n given.");
        }

        $maxOffset = $this->numSamples() - 1;

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
     * @throws InvalidArgumentException
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

        /** @var positive-int $numLevels */
        $numLevels = (int) round(sqrt(count($weights)));

        $levels = array_chunk($weights, $numLevels, true);
        $levelTotals = array_map('array_sum', $levels);

        $total = array_sum($levelTotals);

        $phi = getrandmax() / $total;
        $max = (int) round($total * $phi);

        $samples = [];

        while (count($samples) < $n) {
            $delta = rand(0, $max) / $phi;

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

                        break 2;
                    }
                }
            }
        }

        return self::quick($samples);
    }

    /**
     * Return a row from the dataset at the given offset.
     *
     * @param int $offset
     * @throws InvalidArgumentException
     * @return mixed[]
     */
    #[\ReturnTypeWillChange]
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
     * @return \Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        yield from $this->samples;
    }
}
