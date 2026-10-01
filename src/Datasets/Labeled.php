<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Report;
use Rubix\ML\DataType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Traversable;
use Generator;

use function count;
use function gettype;
use function is_string;
use function is_numeric;
use function is_float;
use function is_nan;
use function array_slice;
use function array_map;
use function array_chunk;
use function array_rand;
use function round;
use function getrandmax;
use function rand;

use function Rubix\ML\linspace;

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
     * @param bool $verify
     * @return self
     */
    public static function fromIterator(iterable $iterator, bool $verify = true) : self
    {
        $samples = $labels = [];

        foreach ($iterator as $record) {
            $labels[] = array_pop($record);
            $samples[] = $record;
        }

        return new self($samples, $labels, $verify);
    }

    /**
     * Build an iterable of datasets of size n from an iterator. The last batch
     * may contain fewer than n samples.
     *
     * @param iterable<mixed[]> $iterator
     * @param int $size
     * @param bool $verify
     * @return Generator<self>
     */
    public static function chunked(iterable $iterator, int $size, bool $verify = true) : Generator
    {
        if ($size < 1) {
            throw new InvalidArgumentException('Chunk size must be greater than 0.');
        }

        $samples = $labels = [];

        foreach ($iterator as $record) {
            $labels[] = array_pop($record);
            $samples[] = $record;

            if (count($samples) === $size) {
                yield new self($samples, $labels, $verify);

                $samples = $labels = [];
            }
        }

        if ($samples) {
            yield new self($samples, $labels, $verify);
        }
    }

    /**
     * Stack a number of datasets on top of each other to form a single dataset.
     *
     * @param iterable<Labeled> $datasets
     * @throws InvalidArgumentException
     * @return self
     */
    public static function stack(iterable $datasets) : self
    {
        $samples = $labels = [];

        foreach ($datasets as $i => $dataset) {
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

            $code = DataType::detectCode($labels[0]);

            if ($code !== DataType::CATEGORICAL and $code !== DataType::CONTINUOUS) {
                throw new InvalidArgumentException('Label type must be'
                    . ' categorical or continuous, ' . DataType::build($code)
                    . ' given.');
            }

            foreach ($labels as $offset => $label) {
                $labelCode = DataType::detectCode($label);

                if ($labelCode !== $code) {
                    throw new InvalidArgumentException('Invalid label type'
                        . " found at offset $offset, " . DataType::build($code)
                        . ' expected but ' . DataType::build($labelCode)
                        . ' given.');
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
    public function label(int $offset) : int|float|string
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

        $samples = $labels = [];

        foreach ($order as $i) {
            $samples[] = $this->samples[$i];
            $labels[] = $this->labels[$i];
        }

        $this->samples = $samples;
        $this->labels = $labels;

        return $this;
    }

    /**
     * Group samples by label and return an array of stratified datasets. i.e.
     * n datasets consisting of samples with the same label where n is equal to
     * the number of unique labels.
     *
     * @return self[]
     */
    public function stratifyByClassLabels() : array
    {
        if (!$this->labelType()->isCategorical()) {
            throw new InvalidArgumentException('Label type must be categorical, '
                . $this->labelType() . ' given.');
        }

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
     * Group samples into equal frequency bins by their continuous label and return
     * an array of binned datasets. Bin edges are derived from the quantiles of the
     * label such that each bin holds a roughly equal number of samples. Bins that
     * contain no samples are dropped from the result.
     *
     * @param int $bins
     * @throws InvalidArgumentException
     * @return self[]
     */
    public function stratifyByLabelBins(int $bins = 10) : array
    {
        if (!$this->labelType()->isContinuous()) {
            throw new InvalidArgumentException('Label type must be continuous, '
                . $this->labelType() . ' given.');
        }

        if ($bins < 1) {
            throw new InvalidArgumentException('The number of bins must be'
                . " greater than 0, $bins given.");
        }

        $bins = min($bins, $this->numSamples());

        /** @var list<float> $values */
        $values = $this->labels;

        $edges = Stats::quantiles(
            $values,
            array_slice(linspace(0.0, 1.0, $bins + 1), 1, -1)
        );

        $numEdges = count($edges);

        $offsets = array_fill(0, $bins, []);

        foreach ($values as $offset => $value) {
            $ordinal = $numEdges;

            foreach ($edges as $j => $edge) {
                if ($value <= $edge) {
                    $ordinal = $j;

                    break;
                }
            }

            $offsets[$ordinal][] = $offset;
        }

        $strata = [];

        foreach ($offsets as $bucket) {
            if (empty($bucket)) {
                continue;
            }

            $samples = $labels = [];

            foreach ($bucket as $offset) {
                $samples[] = $this->samples[$offset];
                $labels[] = $this->labels[$offset];
            }

            $strata[] = self::quick($samples, $labels);
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

        $strata = $this->stratifyByClassLabels();

        $total = (int) floor($ratio * $this->numSamples());

        $quota = $remainder = [];
        $allocated = 0;

        foreach ($strata as $i => $stratum) {
            $base = (int) floor($ratio * $stratum->numSamples());

            $allocated += $base;
            $quota[$i] = $base;
            $remainder[$i] = ($ratio * $stratum->numSamples()) - $base;
        }

        $deficit = $total - $allocated;

        arsort($remainder);

        foreach ($remainder as $i => $share) {
            if ($deficit <= 0) {
                break;
            }

            ++$quota[$i];
            --$deficit;
        }

        $leftStrata = $rightStrata = [];

        foreach ($quota as $i => $count) {
            $stratum = $strata[$i];

            $leftStrata[] = $stratum->slice(0, $count);
            $rightStrata[] = $stratum->slice($count, $stratum->numSamples() - $count);
        }

        return [
            self::stack($leftStrata),
            self::stack($rightStrata),
        ];
    }

    /**
     * Split the dataset into two subsets with a given ratio of samples such that
     * the distribution of the continuous label is preserved in both subsets. The
     * left subset always contains exactly floor($ratio * numSamples()) samples.
     *
     * The number of bins is capped so that the smaller subset can draw at least one
     * sample from every bin. Since bins are ordered by target value, strata are
     * shuffled before the remainder is awarded to prevent ties from consistently
     * favouring the lowest valued bins.
     *
     * @param float $ratio
     * @param int $bins
     * @throws InvalidArgumentException
     * @return array{self,self}
     */
    public function binnedSplit(float $ratio = 0.5, int $bins = 10) : array
    {
        if ($ratio < 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        if ($bins < 1) {
            throw new InvalidArgumentException('Bins must be'
                . " greater than 0, $bins given.");
        }

        $n = $this->numSamples();

        $smallerRatio = min($ratio, 1.0 - $ratio);
        $cap = $smallerRatio > 0.0 ? (int) ceil(1 / $smallerRatio) : max(1, $n);

        $strata = $this->stratifyByLabelBins(max(1, min($bins, intdiv($n, $cap))));

        shuffle($strata);

        $total = (int) floor($ratio * $n);

        $quota = $remainder = [];
        $allocated = 0;

        foreach ($strata as $i => $stratum) {
            $base = (int) floor($ratio * $stratum->numSamples());

            $allocated += $base;
            $quota[$i] = $base;
            $remainder[$i] = ($ratio * $stratum->numSamples()) - $base;
        }

        $deficit = $total - $allocated;

        arsort($remainder);

        foreach ($remainder as $i => $share) {
            if ($deficit <= 0) {
                break;
            }

            ++$quota[$i];
            --$deficit;
        }

        $leftStrata = $rightStrata = [];

        foreach ($quota as $i => $count) {
            $stratum = $strata[$i];

            $leftStrata[] = $stratum->slice(0, $count);
            $rightStrata[] = $stratum->slice($count, $stratum->numSamples() - $count);
        }

        return [
            self::stack($leftStrata),
            self::stack($rightStrata),
        ];
    }

    /**
     * Fold the dataset k - 1 times to form k datasets of as equal size as
     * possible. Any remaining samples are distributed one per fold starting
     * from the first fold, so no two folds ever differ in size by more than
     * a single sample.
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

        if ($k > $this->numSamples()) {
            throw new InvalidArgumentException('K must be less than or equal '
                . 'to the number of samples.');
        }

        $n = intdiv($this->numSamples(), $k);

        $remainder = $this->numSamples() % $k;

        $samples = $this->samples;
        $labels = $this->labels;

        $folds = [];

        for ($j = 0; $j < $k; ++$j) {
            $count = $n + ($j < $remainder ? 1 : 0);

            $folds[] = self::quick(
                array_splice($samples, 0, $count),
                array_splice($labels, 0, $count)
            );
        }

        return $folds;
    }

    /**
     * Fold the dataset into k equal sized stratified datasets. The leftover
     * samples of every stratum are awarded one per fold starting from a cursor
     * that carries over between strata, which keeps the aggregate fold sizes
     * within a single sample of one another instead of piling every remainder
     * into the last fold.
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

        $strata = $this->stratifyByClassLabels();

        foreach ($strata as $stratum) {
            if ($stratum->numSamples() < $k) {
                throw new InvalidArgumentException('K must be less than or '
                    . 'equal to the number of samples in the smallest '
                    . 'stratum.');
            }
        }

        $folds = array_fill(0, $k, []);

        $cursor = 0;

        foreach ($strata as $stratum) {
            $m = $stratum->numSamples();

            $n = intdiv($m, $k);
            $remainder = $m % $k;

            $offset = 0;

            for ($j = 0; $j < $k; ++$j) {
                $count = $n + ((($j - $cursor + $k) % $k) < $remainder ? 1 : 0);

                $folds[$j][] = $stratum->slice($offset, $count);

                $offset += $count;
            }

            $cursor = ($cursor + $remainder) % $k;
        }

        foreach ($folds as &$fold) {
            $fold = self::stack($fold);
        }

        unset($fold);

        /** @var list<self> $folds */
        return $folds;
    }

    /**
     * Fold the dataset into k equal sized datasets such that the distribution of
     * the continuous label is preserved in every fold. The leftover samples of
     * every bin are awarded one per fold starting from a cursor that carries over
     * between bins, which keeps the aggregate fold sizes within a single sample
     * of one another instead of piling every remainder into the last fold.
     *
     * @param int $k
     * @param int $bins
     * @throws InvalidArgumentException
     * @return list<self>
     */
    public function binnedFold(int $k = 10, int $bins = 10) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot create less than'
                . " 2 folds, $k given.");
        }

        if ($bins < 1) {
            throw new InvalidArgumentException('Bins must be'
                . " greater than 0, $bins given.");
        }

        $bins = max(1, min($bins, intdiv($this->numSamples(), $k)));

        $strata = $this->stratifyByLabelBins($bins);

        foreach ($strata as $stratum) {
            if ($stratum->numSamples() < $k) {
                throw new InvalidArgumentException('K must be less than or '
                    . 'equal to the number of samples in the smallest '
                    . 'bin.');
            }
        }

        $folds = array_fill(0, $k, []);

        $cursor = 0;

        foreach ($strata as $stratum) {
            $m = $stratum->numSamples();

            $n = intdiv($m, $k);
            $remainder = $m % $k;

            $offset = 0;

            for ($j = 0; $j < $k; ++$j) {
                $count = $n + ((($j - $cursor + $k) % $k) < $remainder ? 1 : 0);

                $folds[$j][] = $stratum->slice($offset, $count);

                $offset += $count;
            }

            $cursor = ($cursor + $remainder) % $k;
        }

        foreach ($folds as &$fold) {
            $fold = self::stack($fold);
        }

        unset($fold);

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
    public function splitByFeature(int $column, string|int|float $value) : array
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
        if ($this->empty()) {
            throw new InvalidArgumentException('Cannot generate'
                . ' a random subset from an empty dataset.');
        }

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
        if ($this->empty()) {
            throw new InvalidArgumentException('Cannot generate'
                . ' a random subset from an empty dataset.');
        }

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

        $total = 0.0;
        $cums = [];

        foreach ($weights as $weight) {
            $total += $weight;

            $cums[] = $total;
        }

        /** @var positive-int $numWeights */
        $numWeights = count($cums);

        $phi = getrandmax() / $total;
        $max = (int) round($total * $phi);

        $samples = $labels = [];

        while (count($samples) < $n) {
            $delta = rand(0, $max) / $phi;

            $lower = 0;
            $upper = $numWeights - 1;

            while ($lower < $upper) {
                $mid = intdiv($lower + $upper, 2);

                if ($cums[$mid] < $delta) {
                    $lower = ++$mid;
                } else {
                    $upper = $mid;
                }
            }

            $samples[] = $this->samples[$lower];
            $labels[] = $this->labels[$lower];
        }

        return self::quick($samples, $labels);
    }

    /**
     * Describe the features of the dataset broken down by label.
     *
     * @return Report
     */
    public function describeByClassLabels() : Report
    {
        $stats = [];

        foreach ($this->stratifyByClassLabels() as $label => $stratum) {
            $stats[$label] = $stratum->describe()->toArray();
        }

        return new Report($stats);
    }

    /**
     * Describe the features of the dataset broken down by label bin. Bins are
     * equal frequency bins derived from the quantiles of the continuous label
     * and the report is keyed by bin ordinal in ascending order of target value.
     *
     * @param int $bins
     * @throws InvalidArgumentException
     * @return Report
     */
    public function describeByLabelBins(int $bins = 10) : Report
    {
        $stats = [];

        foreach ($this->stratifyByLabelBins($bins) as $stratum) {
            $stats[] = $stratum->describe()->toArray();
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
     * @return Generator<mixed[]>
     */
    public function getIterator() : Traversable
    {
        foreach ($this->samples as $i => $sample) {
            $sample[] = $this->labels[$i];

            yield $sample;
        }
    }
}
