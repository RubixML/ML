<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Report;
use Rubix\ML\DataType;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Traversable;

use function count;
use function get_class;
use function gettype;
use function is_string;
use function is_numeric;
use function is_float;
use function is_nan;
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
 * Labeled
 *
 * A Labeled dataset is used to train supervised learners and for testing a model by
 * providing the ground-truth. In addition to the standard dataset object methods, a
 * Labeled dataset can perform operations such as stratification and sorting the
 * dataset by label.
 *
 * > **Note:** Labels can be of categorical or continuous data type but NaN values
 * are not allowed.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Labeled extends Dataset
{
    /**
     * The observed outcomes for each sample in the dataset.
     *
     * @var list<int|float|string>
     */
    protected array $labels;

    /**
     * Build a new labeled dataset with validation.
     *
     * @param array<mixed[]> $samples
     * @param (string|int|float)[] $labels
     * @return self
     */
    public static function build(array $samples = [], array $labels = []) : self
    {
        return new self($samples, $labels, true);
    }

    /**
     * Build a new labeled dataset foregoing validation.
     *
     * @param array<mixed[]> $samples
     * @param (string|int|float)[] $labels
     * @return self
     */
    public static function quick(array $samples = [], array $labels = []) : self
    {
        return new self($samples, $labels, false);
    }

    /**
     * Build a dataset with the rows from an iterable data table.
     *
     * @param iterable<mixed[]> $iterator
     * @return self
     */
    public static function fromIterator(iterable $iterator) : self
    {
        $samples = $labels = [];

        foreach ($iterator as $record) {
            $labels[] = array_pop($record);
            $samples[] = $record;
        }

        return self::build($samples, $labels);
    }

    /**
     * Stack a number of datasets on top of each other to form a single dataset.
     *
     * @param iterable<\Rubix\ML\Datasets\Labeled> $datasets
     * @throws InvalidArgumentException
     * @return self
     */
    public static function stack(iterable $datasets) : self
    {
        $samples = $labels = [];

        foreach ($datasets as $i => $dataset) {
            if (!$dataset instanceof Labeled) {
                throw new InvalidArgumentException('Dataset must be'
                    . ' an instance of Labeled, ' . get_class($dataset)
                    . ' given.');
            }

            if ($dataset->empty()) {
                continue;
            }

            if (isset($lastNumFeatures) and $dataset->numFeatures() !== $lastNumFeatures) {
                throw new InvalidArgumentException("Dataset $i must have"
                    . " the same number of columns, $lastNumFeatures"
                    . " expected but {$dataset->numFeatures()} given.");
            }

            $samples[] = $dataset->samples();
            $labels[] = $dataset->labels();

            $lastNumFeatures = $dataset->numFeatures();
        }

        return self::quick(
            array_merge(...$samples),
            array_merge(...$labels)
        );
    }

    /**
     * @param array<mixed[]> $samples
     * @param (string|int|float)[] $labels
     * @param bool $verify
     * @throws InvalidArgumentException
     */
    public function __construct(array $samples = [], array $labels = [], bool $verify = true)
    {
        if (count($samples) !== count($labels)) {
            throw new InvalidArgumentException('Number of samples'
             . ' and labels must be equal, ' . count($samples)
             . ' samples but ' . count($labels) . ' labels given.');
        }

        if ($verify and $labels) {
            $labels = array_values($labels);

            $type = DataType::detect($labels[0]);

            if (!$type->isCategorical() and !$type->isContinuous()) {
                throw new InvalidArgumentException('Label type must be'
                    . " categorical or continuous, $type given.");
            }

            foreach ($labels as $offset => $label) {
                if (DataType::detect($label) != $type) {
                    throw new InvalidArgumentException('Invalid label type'
                        . " found at offset $offset, $type expected but "
                        . DataType::detect($label) . ' given.');
                }

                if (is_float($label) and is_nan($label)) {
                    throw new InvalidArgumentException('Labels must not'
                        . " contain NaN values, NaN found at offset $offset.");
                }
            }
        }

        $this->labels = $labels;

        parent::__construct($samples, $verify);
    }

    /**
     * Return the labels.
     *
     * @return mixed[]
     */
    public function labels() : array
    {
        return $this->labels;
    }

    /**
     * Return a label at the given row offset.
     *
     * @param int $offset
     * @throws InvalidArgumentException
     * @return int|float|string
     */
    public function label(int $offset)
    {
        if (!isset($this->labels[$offset])) {
            throw new InvalidArgumentException("Row at offset $offset not found.");
        }

        return $this->labels[$offset];
    }

    /**
     * Return the integer encoded data type of the label or null if empty.
     *
     * @throws RuntimeException
     * @return DataType
     */
    public function labelType() : DataType
    {
        if (empty($this->labels)) {
            throw new RuntimeException('Dataset is empty.');
        }

        return DataType::detect(current($this->labels));
    }

    /**
     * Map labels to their new values and return self for method chaining.
     *
     * @param callable $callback
     * @throws RuntimeException
     * @return self
     */
    public function transformLabels(callable $callback) : self
    {
        $labels = array_map($callback, $this->labels);

        foreach ($labels as $label) {
            if (!is_string($label) and !is_numeric($label)) {
                throw new RuntimeException('Label must be a string or'
                    . ' numeric type, ' . gettype($label) . ' found.');
            }
        }

        $this->labels = $labels;

        return $this;
    }

    /**
     * The set of all possible labels.
     *
     * @return mixed[]
     */
    public function possibleOutcomes() : array
    {
        return array_values(array_unique($this->labels));
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
     * Take n samples and labels from this dataset and return them in a new
     * dataset.
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
     * Leave n samples and labels on this dataset and return the rest in a new
     * dataset.
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
        return self::quick(
            array_slice($this->samples, $offset, $n),
            array_slice($this->labels, $offset, $n)
        );
    }

    /**
     * Remove a size n chunk of the dataset starting at offset and return it in a new dataset.
     *
     * @param int $offset
     * @param int $n
     * @return self
     */
    public function splice(int $offset, int $n) : self
    {
        return self::quick(
            array_splice($this->samples, $offset, $n),
            array_splice($this->labels, $offset, $n)
        );
    }

    /**
     * Merge the rows of this dataset with another dataset.
     *
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     * @return self
     */
    public function merge(Dataset $dataset) : self
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Can only merge'
                . ' with another Labeled dataset.');
        }

        if (!$dataset->empty() and !$this->empty()) {
            if ($dataset->numFeatures() !== $this->numFeatures()) {
                throw new InvalidArgumentException('Datasets must have'
                    . " the same number of columns, {$this->numFeatures()}"
                    . " expected, but {$dataset->numFeatures()} given.");
            }
        }

        return self::quick(
            array_merge($this->samples, $dataset->samples()),
            array_merge($this->labels, $dataset->labels())
        );
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

        return self::quick($samples, $this->labels);
    }

    /**
     * Randomize the dataset in place and return self for chaining.
     *
     * @return self
     */
    public function randomize() : self
    {
        if ($this->empty()) {
            return $this;
        }

        $order = range(0, $this->numSamples() - 1);

        shuffle($order);

        array_multisort($order, $this->samples, $this->labels);

        return $this;
    }

    /**
     * Group samples by label and return an array of stratified datasets. i.e.
     * n datasets consisting of samples with the same label where n is equal to
     * the number of unique labels.
     *
     * @return self[]
     */
    public function stratifyByLabel() : array
    {
        $strata = [];

        foreach ($this->labels as $i => $label) {
            $strata[$label][] = $this->samples[$i];
        }

        foreach ($strata as $label => &$stratum) {
            $labels = array_fill(0, count($stratum), $label);

            $stratum = self::quick($stratum, $labels);
        }

        /** @var self[] $strata */
        return $strata;
    }

    /**
     * Split the dataset into two subsets with a given ratio of samples.
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

        $left = self::quick(
            array_slice($this->samples, 0, $n),
            array_slice($this->labels, 0, $n)
        );

        $right = self::quick(
            array_slice($this->samples, $n),
            array_slice($this->labels, $n)
        );

        return [$left, $right];
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param float $ratio
     * @throws InvalidArgumentException
     * @return array{self,self}
     */
    public function stratifiedSplit(float $ratio = 0.5) : array
    {
        if ($ratio < 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        $leftStrata = $rightStrata = [];

        foreach ($this->stratifyByLabel() as $stratum) {
            [$left, $right] = $stratum->split($ratio);

            $leftStrata[] = $left;
            $rightStrata[] = $right;
        }

        $left = self::stack($leftStrata);
        $right = self::stack($rightStrata);

        return [$left, $right];
    }

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param int $k
     * @throws InvalidArgumentException
     * @return list<self>
     */
    public function fold(int $k = 10) : array
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Cannot create less than'
                . " 1 fold, $k given.");
        }

        $n = (int) floor($this->numSamples() / $k);

        $samples = $this->samples;
        $labels = $this->labels;

        $folds = [];

        while (count($folds) < $k) {
            $folds[] = self::quick(
                array_splice($samples, 0, $n),
                array_splice($labels, 0, $n)
            );
        }

        return $folds;
    }

    /**
     * Fold the dataset into k equal sized stratified datasets.
     *
     * @param int $k
     * @throws InvalidArgumentException
     * @return list<self>
     */
    public function stratifiedFold(int $k = 10) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot create less than'
                . " 2 folds, $k given.");
        }

        $folds = [];

        foreach ($this->stratifyByLabel() as $stratum) {
            foreach ($stratum->fold($k) as $j => $fold) {
                $folds[$j][] = $fold;
            }
        }

        foreach ($folds as &$fold) {
            $fold = self::stack($fold);
        }

        /** @var list<self> $folds */
        return $folds;
    }

    /**
     * Generate a collection of batches of size n from the dataset. If there are
     * not enough samples to fill an entire batch, then the dataset will contain
     * as many samples and labels as possible.
     *
     * @param positive-int $n
     * @return list<self>
     */
    public function batch(int $n = 50) : array
    {
        return array_map(
            [self::class, 'quick'],
            array_chunk($this->samples, $n),
            array_chunk($this->labels, $n)
        );
    }

    /**
     * Split the dataset into left and right subsets using the values of a single feature column for comparison.
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
        $type = $this->featureType($column);

        $leftSamples = $leftLabels = $rightSamples = $rightLabels = [];

        if ($type->isContinuous()) {
            foreach ($this->samples as $i => $sample) {
                if ($sample[$column] <= $value) {
                    $leftSamples[] = $sample;
                    $leftLabels[] = $this->labels[$i];
                } else {
                    $rightSamples[] = $sample;
                    $rightLabels[] = $this->labels[$i];
                }
            }
        } else {
            foreach ($this->samples as $i => $sample) {
                if ($sample[$column] === $value) {
                    $leftSamples[] = $sample;
                    $leftLabels[] = $this->labels[$i];
                } else {
                    $rightSamples[] = $sample;
                    $rightLabels[] = $this->labels[$i];
                }
            }
        }

        return [
            self::quick($leftSamples, $leftLabels),
            self::quick($rightSamples, $rightLabels),
        ];
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
        $leftSamples = $leftLabels = $rightSamples = $rightLabels = [];

        foreach ($this->samples as $i => $sample) {
            $lDistance = $kernel->compute($sample, $leftCentroid);
            $rDistance = $kernel->compute($sample, $rightCentroid);

            if ($lDistance < $rDistance) {
                $leftSamples[] = $sample;
                $leftLabels[] = $this->labels[$i];
            } else {
                $rightSamples[] = $sample;
                $rightLabels[] = $this->labels[$i];
            }
        }

        return [
            self::quick($leftSamples, $leftLabels),
            self::quick($rightSamples, $rightLabels),
        ];
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
            throw new InvalidArgumentException('Cannot generate subset'
                . " of less than 1 sample, $n given.");
        }

        if ($n > $this->numSamples()) {
            throw new InvalidArgumentException('Cannot generate subset'
                . " of more than {$this->numSamples()}, $n given.");
        }

        $offsets = array_rand($this->samples, $n);

        $offsets = is_array($offsets) ? $offsets : [$offsets];

        $samples = $labels = [];

        foreach ($offsets as $offset) {
            $samples[] = $this->samples[$offset];
            $labels[] = $this->labels[$offset];
        }

        return self::quick($samples, $labels);
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
            throw new InvalidArgumentException('Cannot generate'
                . " subset of less than 1 sample, $n given.");
        }

        $maxOffset = $this->numSamples() - 1;

        $samples = $labels = [];

        while (count($samples) < $n) {
            $offset = rand(0, $maxOffset);

            $samples[] = $this->samples[$offset];
            $labels[] = $this->labels[$offset];
        }

        return self::quick($samples, $labels);
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
            throw new InvalidArgumentException('Cannot generate'
                . " subset of less than 1 sample, $n given.");
        }

        if (count($weights) !== count($this->samples)) {
            throw new InvalidArgumentException('The number of weights'
                . ' must be equal to the number of samples in the'
                . ' dataset, ' . count($this->samples) . ' needed'
                . ' but ' . count($weights) . ' given.');
        }

        /** @var positive-int $numLevels */
        $numLevels = (int) round(sqrt(count($weights))) ?: 1;

        $levels = array_chunk($weights, $numLevels, true);

        $levelTotals = array_map('array_sum', $levels);

        $total = array_sum($levelTotals);

        $phi = getrandmax() / $total;
        $max = (int) round($total * $phi);

        $samples = $labels = [];

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
                        $labels[] = $this->labels[$offset];

                        break 2;
                    }
                }
            }
        }

        return self::quick($samples, $labels);
    }

    /**
     * Describe the features of the dataset broken down by label.
     *
     * @return Report
     */
    public function describeByLabel() : Report
    {
        $stats = [];

        foreach ($this->stratifyByLabel() as $label => $stratum) {
            $stats[$label] = $stratum->describe()->toArray();
        }

        return new Report($stats);
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
            return array_merge($this->samples[$offset], [$this->labels[$offset]]);
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
        foreach ($this->samples as $i => $sample) {
            $sample[] = $this->labels[$i];

            yield $sample;
        }
    }
}
