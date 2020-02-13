<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\DataType;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Console;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithDistance;
use InvalidArgumentException;
use RuntimeException;
use ErrorException;
use Generator;

use function count;
use function get_class;
use function gettype;
use function array_slice;

use const Rubix\ML\PHI;
use const Rubix\ML\EPSILON;

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
     * @var (int|float|string)[]
     */
    protected $labels;

    /**
     * Build a new labeled dataset with validation.
     *
     * @param array[] $samples
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
     * @param array[] $samples
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
     * @param iterable<array> $iterator
     * @return self
     */
    public static function fromIterator(iterable $iterator) : self
    {
        $samples = $labels = [];

        foreach ($iterator as $record) {
            $samples[] = array_slice($record, 0, -1);
            $labels[] = end($record);
        }

        return self::build($samples, $labels);
    }

    /**
     * Stack a number of datasets on top of each other to form a single
     * dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled[] $datasets
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function stack(array $datasets) : self
    {
        $samples = $labels = [];

        foreach ($datasets as $dataset) {
            if (!$dataset instanceof Labeled) {
                throw new InvalidArgumentException('Dataset must be'
                    . ' an instance of Labeled, ' . get_class($dataset)
                    . ' given.');
            }

            $samples[] = $dataset->samples();
            $labels[] = $dataset->labels();
        }

        return self::quick(
            array_merge(...$samples),
            array_merge(...$labels)
        );
    }

    /**
     * @param array[] $samples
     * @param (string|int|float)[] $labels
     * @param bool $validate
     * @throws \InvalidArgumentException
     */
    public function __construct(array $samples = [], array $labels = [], bool $validate = true)
    {
        if (count($samples) !== count($labels)) {
            throw new InvalidArgumentException('The proportion of samples to'
             . ' labels must be equal, ' . count($samples) . ' samples and '
             . count($labels) . ' labels given.');
        }

        if ($validate and $labels) {
            $labels = array_values($labels);

            $type = DataType::determine($labels[0]);

            if (!$type->isCategorical() and !$type->isContinuous()) {
                throw new InvalidArgumentException('Label type must be'
                    . " categorical or continuous, $type given.");
            }

            foreach ($labels as $label) {
                if (DataType::determine($label) != $type) {
                    throw new InvalidArgumentException('Labels must all be'
                        . " the same high-level data type, expected $type"
                        . ' but ' . DataType::determine($label) . ' given.');
                }

                if (is_float($label) and is_nan($label)) {
                    throw new InvalidArgumentException('Labels must not'
                        . ' contain NaN values.');
                }
            }
        }

        $this->labels = $labels;

        parent::__construct($samples, $validate);
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
     * @throws \InvalidArgumentException
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
     * @throws \RuntimeException
     * @return \Rubix\ML\DataType
     */
    public function labelType() : DataType
    {
        if (empty($this->labels)) {
            throw new RuntimeException('Dataset is empty.');
        }

        return DataType::determine(current($this->labels));
    }

    /**
     * Map labels to their new values and return self for method chaining.
     *
     * @param callable $callback
     * @throws \RuntimeException
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
     * Describe the features of the dataset broken down by label.
     *
     * @return mixed[]
     */
    public function describeByLabel() : array
    {
        $stats = [];

        foreach ($this->stratify() as $label => $stratum) {
            $stats[$label] = $stratum->describe();
        }

        return $stats;
    }

    /**
     * Return an array of descriptive statistics about the labels in the
     * dataset.
     *
     * @return mixed[]
     */
    public function describeLabels() : array
    {
        $type = $this->labelType();

        $desc = [];
        
        $desc['type'] = (string) $type;

        switch ($type) {
            case DataType::continuous():
                [$mean, $variance] = Stats::meanVar($this->labels);

                $desc['mean'] = $mean;
                $desc['variance'] = $variance;
                $desc['std_dev'] = sqrt($variance ?: EPSILON);
                $desc['skewness'] = Stats::skewness($this->labels, $mean);
                $desc['kurtosis'] = Stats::kurtosis($this->labels, $mean);

                $percentiles = Stats::percentiles($this->labels, [
                    0, 25, 50, 75, 100,
                ]);

                $desc['min'] = $percentiles[0];
                $desc['25%'] = $percentiles[1];
                $desc['median'] = $percentiles[2];
                $desc['75%'] = $percentiles[3];
                $desc['max'] = $percentiles[4];

                break 1;

            case DataType::categorical():
                $counts = array_count_values($this->labels);

                $total = count($this->labels) ?: EPSILON;

                $probabilities = [];

                foreach ($counts as $class => $count) {
                    $probabilities[$class] = $count / $total;
                }

                $desc['num_categories'] = count($counts);
                $desc['probabilities'] = $probabilities;

                break 1;
        }

        return $desc;
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
     * Take n samples and labels from this dataset and return them in a new
     * dataset.
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
     * Leave n samples and labels on this dataset and return the rest in a new
     * dataset.
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
        return self::quick(
            array_slice($this->samples, $offset, $n),
            array_slice($this->labels, $offset, $n)
        );
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
        return self::quick(
            array_splice($this->samples, $offset, $n),
            array_splice($this->labels, $offset, $n)
        );
    }

    /**
     * Prepend a dataset to this dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @return self
     */
    public function prepend(Dataset $dataset) : self
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Can only prepend with another'
                . ' labeled dataset.');
        }

        if ((!$dataset->empty() and !$this->empty()) and $dataset->numColumns() !== $this->numColumns()) {
            throw new InvalidArgumentException('Can only append with dataset'
                . ' that has the same number of columns.');
        }

        return self::quick(
            array_merge($dataset->samples(), $this->samples),
            array_merge($dataset->labels(), $this->labels)
        );
    }

    /**
     * Append a dataset to this dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @return self
     */
    public function append(Dataset $dataset) : self
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Can only append with another'
                . ' labeled dataset.');
        }

        if ((!$dataset->empty() and !$this->empty()) and $dataset->numColumns() !== $this->numColumns()) {
            throw new InvalidArgumentException('Can only append with dataset'
                . ' that has the same number of columns.');
        }

        return self::quick(
            array_merge($this->samples, $dataset->samples()),
            array_merge($this->labels, $dataset->labels())
        );
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
     * @throws \InvalidArgumentException
     * @return self
     */
    public function dropRows(array $offsets) : self
    {
        foreach ($offsets as $offset) {
            unset($this->samples[$offset], $this->labels[$offset]);
        }

        $this->samples = array_values($this->samples);
        $this->labels = array_values($this->labels);

        return $this;
    }

    /**
     * Randomize the dataset in place and return self for chaining.
     *
     * @return self
     */
    public function randomize() : self
    {
        $order = range(0, max(0, $this->numRows() - 1));

        shuffle($order);

        array_multisort($order, $this->samples, $this->labels);

        return $this;
    }

    /**
     * Filter the rows of the dataset using the values of a feature column at the given
     * offset as the arguments to a filter callback. The callback should return false
     * for rows that should be filtered.
     *
     * @param int $offset
     * @param callable $callback
     * @return self
     */
    public function filterByColumn(int $offset, callable $callback) : self
    {
        $samples = $labels = [];

        foreach ($this->samples as $i => $sample) {
            if ($callback($sample[$offset])) {
                $samples[] = $sample;
                $labels[] = $this->labels[$i];
            }
        }

        return self::quick($samples, $labels);
    }

    /**
     * Filter the rows of the dataset using the labels as the argument to a callback.
     *
     * @param callable $callback
     * @return self
     */
    public function filterByLabel(callable $callback) : self
    {
        $samples = $labels = [];

        foreach ($this->labels as $i => $label) {
            if ($callback($label)) {
                $samples[] = $this->samples[$i];
                $labels[] = $label;
            }
        }

        return self::quick($samples, $labels);
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
        $order = $this->column($offset);

        array_multisort(
            $order,
            $this->samples,
            $this->labels,
            $descending ? SORT_DESC : SORT_ASC
        );

        return $this;
    }

    /**
     * Sort the dataset in place by its labels.
     *
     * @param bool $descending
     * @return self
     */
    public function sortByLabel(bool $descending = false) : self
    {
        array_multisort(
            $this->labels,
            $this->samples,
            $descending ? SORT_DESC : SORT_ASC
        );

        return $this;
    }

    /**
     * Group samples by label and return an array of stratified datasets. i.e.
     * n datasets consisting of samples with the same label where n is equal to
     * the number of unique labels.
     *
     * @return self[]
     */
    public function stratify() : array
    {
        $strata = [];

        foreach ($this->_stratify() as $label => $stratum) {
            $labels = array_fill(0, count($stratum), $label);

            $strata[$label] = self::quick($stratum, $labels);
        }

        return $strata;
    }

    /**
     * Split the dataset into two subsets with a given ratio of samples.
     *
     * @param float $ratio
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Ratio must be strictly'
                . " between 0 and 1, $ratio given.");
        }

        $n = (int) floor($ratio * $this->numRows());

        return [
            self::quick(
                array_slice($this->samples, 0, $n),
                array_slice($this->labels, 0, $n)
            ),
            self::quick(
                array_slice($this->samples, $n),
                array_slice($this->labels, $n)
            ),
        ];
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param float $ratio
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function stratifiedSplit(float $ratio = 0.5) : array
    {
        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Ratio must be strictly'
                . " between 0 and 1, $ratio given.");
        }

        $leftSamples = $leftLabels = $rightSamples = $rightLabels = [];

        foreach ($this->_stratify() as $label => $stratum) {
            $n = (int) floor($ratio * count($stratum));

            $leftSamples[] = array_splice($stratum, 0, $n);
            $leftLabels[] = array_fill(0, $n, $label);

            $rightSamples[] = $stratum;
            $rightLabels[] = array_fill(0, count($stratum), $label);
        }

        return [
            self::quick(
                array_merge(...$leftSamples),
                array_merge(...$leftLabels)
            ),
            self::quick(
                array_merge(...$rightSamples),
                array_merge(...$rightLabels)
            ),
        ];
    }

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param int $k
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function fold(int $k = 10) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot create less than'
                . " 2 folds, $k given.");
        }

        $n = (int) floor($this->numRows() / $k);

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
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function stratifiedFold(int $k = 10) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot create less than'
                . " 2 folds, $k given.");
        }

        $folds = [];

        for ($i = 0; $i < $k; ++$i) {
            $samples = $labels = [];

            foreach ($this->_stratify() as $label => $stratum) {
                $n = (int) floor(count($stratum) / $k);

                $samples[] = array_slice($stratum, $i * $n, $n);
                $labels[] = array_fill(0, $n, $label);
            }

            $folds[] = self::quick(
                array_merge(...$samples),
                array_merge(...$labels)
            );
        }

        return $folds;
    }

    /**
     * Stratifying subroutine groups samples by their categorical label.
     *
     * @throws \RuntimeException
     * @return array[]
     */
    protected function _stratify() : array
    {
        $strata = [];

        try {
            foreach ($this->labels as $i => $label) {
                $strata[$label][] = $this->samples[$i];
            }
        } catch (ErrorException $e) {
            throw new RuntimeException('Label must be an integer or string.');
        }

        return $strata;
    }

    /**
     * Generate a collection of batches of size n from the dataset. If there are
     * not enough samples to fill an entire batch, then the dataset will contain
     * as many samples and labels as possible.
     *
     * @param int $n
     * @return self[]
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
     * Partition the dataset into left and right subsets by a specified feature
     * column. The dataset is split such that, for categorical values, the left
     * subset contains all samples that match the value and the right side
     * contains samples that do not match. For continuous values, the left side
     * contains all the  samples that are less than the target value, and the
     * right side contains the samples that are greater than or equal to the
     * value.
     *
     * @param int $column
     * @param string|int|float $value
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function partition(int $column, $value) : array
    {
        $leftSamples = $leftLabels = $rightSamples = $rightLabels = [];

        if ($this->columnType($column)->isContinuous()) {
            foreach ($this->samples as $i => $sample) {
                if ($sample[$column] < $value) {
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
     * Partition the dataset into left and right subsets based on their distance
     * from two centroids.
     *
     * @param (string|int|float)[] $leftCentroid
     * @param (string|int|float)[] $rightCentroid
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function spatialPartition(array $leftCentroid, array $rightCentroid, Distance $kernel)
    {
        if (count($leftCentroid) !== count($rightCentroid)) {
            throw new InvalidArgumentException('Dimensionality mismatch between'
                . ' left and right centroids.');
        }

        SamplesAreCompatibleWithDistance::check($this, $kernel);

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
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubset(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate subset'
                . " of less than 1 sample, $n given.");
        }

        if ($n > $this->numRows()) {
            throw new InvalidArgumentException('Cannot generate subset'
                . " of more than {$this->numRows()}, $n given.");
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
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubsetWithReplacement(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate'
                . " subset of less than 1 sample, $n given.");
        }

        $maxOffset = $this->numRows() - 1;

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
     * @throws \InvalidArgumentException
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

        $total = array_sum($weights);
        $max = (int) round($total * PHI);

        $samples = $labels = [];

        while (count($samples) < $n) {
            $delta = rand(0, $max) / PHI;

            foreach ($weights as $offset => $weight) {
                $delta -= $weight;

                if ($delta <= 0.0) {
                    $samples[] = $this->samples[$offset];
                    $labels[] = $this->labels[$offset];

                    break 1;
                }
            }
        }

        return self::quick($samples, $labels);
    }

    /**
     * Remove duplicate rows from the dataset.
     *
     * @return self
     */
    public function deduplicate() : self
    {
        $table = array_unique($this->toArray(), SORT_REGULAR);
        
        $this->samples = array_values(array_intersect_key($this->samples, $table));
        $this->labels = array_values(array_intersect_key($this->labels, $table));
        
        return $this;
    }

    /**
     * Return the dataset object as a data table array.
     *
     * @return array[]
     */
    public function toArray() : array
    {
        return iterator_to_array($this->getIterator());
    }

    /**
     * Return a row from the dataset at the given offset.
     *
     * @param mixed $offset
     * @throws \InvalidArgumentException
     * @return mixed[]
     */
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
     * @return \Generator<array>
     */
    public function getIterator() : Generator
    {
        foreach ($this->samples as $i => $sample) {
            $sample[] = $this->labels[$i];

            yield $sample;
        }
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
        $n = (int) floor($tCols / (3 + Console::TABLE_CELL_WIDTH)) - 1;

        $m = min($this->numRows(), $m);
        $n = min($this->numColumns(), $n);

        $header = [];

        for ($column = 0; $column < $n; ++$column) {
            $header[] = "Column $column";
        }

        $header[] = 'Label';

        $table = array_slice($this->samples, 0, $m);

        foreach ($table as $i => &$row) {
            $row = array_slice($row, 0, $n);

            $row[] = $this->labels[$i];
        }

        array_unshift($table, $header);

        return Console::table($table);
    }
}
