<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Other\Helpers\DataType;
use InvalidArgumentException;
use RuntimeException;

/**
 * Labeled
 *
 * For *supervised* Estimators you will need to train it with a Labeled dataset
 * consisting of a sample matrix with the addition of an array of labels that
 * correspond to the observed outcome of each sample. Splitting, folding,
 * randomizing, sorting, and subsampling are all done while keeping the indices
 * of samples and labels aligned.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Labeled extends DataFrame implements Dataset
{
    /**
     * The observed outcomes for each sample in the dataset.
     *
     * @var (int|float|string)[]
     */
    protected $labels = [
        //
    ];

    /**
     * Build a new labeled dataset with validation.
     *
     * @param  array  $samples
     * @param  array  $labels
     * @return self
     */
    public static function build(array $samples = [], array $labels = []) : self
    {
        return new self($samples, $labels, true);
    }

    /**
     * Build a new labeled dataset foregoing validation.
     *
     * @param  array[]  $samples
     * @param  (int|float|string)[]  $labels
     * @return self
     */
    public static function quick(array $samples = [], array $labels = []) : self
    {
        return new self($samples, $labels, false);
    }

    /**
     * Build a dataset from an iterator.
     *
     * @param  iterable  $samples
     * @param  iterable  $labels
     * @return self
     */
    public static function fromIterator(iterable $samples, iterable $labels) : self
    {
        $samples = is_array($samples)
            ? $samples
            : iterator_to_array($samples, false);


        $labels = is_array($labels)
            ? $labels
            : iterator_to_array($labels, false);

        return self::build($samples, $labels);
    }

    /**
     * Stack a number of datasets on top of each other to form a single
     * dataset.
     *
     * @param  array  $datasets
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function stack(array $datasets) : self
    {
        $samples = $labels = [];

        foreach ($datasets as $dataset) {
            if (!$dataset instanceof self) {
                throw new InvalidArgumentException('Dataset must be'
                    . ' an instance of Labeled, ' . get_class($dataset)
                    . ' given.');
            }

            $samples = array_merge($samples, $dataset->samples());
            $labels = array_merge($labels, $dataset->labels());
        }

        return self::quick($samples, $labels);
    }

    /**
     * @param  array  $samples
     * @param  array  $labels
     * @param  bool  $validate
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $samples = [], array $labels = [], bool $validate = true)
    {
        if (count($samples) !== count($labels)) {
            throw new InvalidArgumentException('The proportion of samples to'
             . ' labels must be equal, ' . count($samples) . ' samples and '
             . count($labels) . ' labels given.');
        }

        if ($validate) {
            $labels = array_values($labels);

            foreach ($labels as $label) {
                if (!is_string($label) and !is_numeric($label)) {
                    throw new InvalidArgumentException('Label must be a string'
                        . ' or numeric type, ' . gettype($label) . ' found.');
                }
            }
        }

        $this->labels = $labels;

        parent::__construct($samples, $validate);
    }

    /**
     * Return the labels.
     *
     * @return (int|float|string)[]
     */
    public function labels() : array
    {
        return $this->labels;
    }

    /**
     * Return the samples and labels in a single array.
     *
     * @return array[]
     */
    public function zip() : array
    {
        $rows = $this->samples;

        foreach ($rows as $i => &$row) {
            $row[] = $this->labels[$i];
        }

        return $rows;
    }

    /**
     * Return a label given by row index.
     *
     * @param  int  $index
     * @throws \InvalidArgumentException
     * @return int|float|string
     */
    public function label(int $index)
    {
        if (!isset($this->labels[$index])) {
            throw new InvalidArgumentException("Row at offset $index"
                . ' does not exist.');
        }

        return $this->labels[$index];
    }

    /**
     * Return the integer encoded data type of the label or null if empty.
     *
     * @return int|null
     */
    public function labelType() : ?int
    {
        if (!isset($this->labels[0])) {
            return null;
        }

        return DataType::determine($this->labels[0]);
    }

    /**
     * Map labels to their new values.
     *
     * @param  callable  $fn
     * @throws \RuntimeException
     * @return void
     */
    public function transformLabels(callable $fn) : void
    {
        $labels = array_map($fn, $this->labels);

        foreach ($labels as $label) {
            if (!is_string($label) and !is_numeric($label)) {
                throw new RuntimeException('Label must be a string or'
                    . ' numeric type, ' . gettype($label) . ' found.');
            }
        }

        $this->labels = $labels;
    }

    /**
     * The set of all possible labeled outcomes.
     *
     * @return array
     */
    public function possibleOutcomes() : array
    {
        return array_values(array_unique($this->labels, SORT_REGULAR));
    }

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function head(int $n = 10) : self
    {
        $samples = array_slice($this->samples, 0, $n);
        $labels = array_slice($this->labels, 0, $n);

        return self::quick($samples, $labels);
    }

    /**
     * Return a dataset containing only the last n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function tail(int $n = 10) : self
    {
        $samples = array_slice($this->samples, -$n);
        $labels = array_slice($this->labels, -$n);

        return self::quick($samples, $labels);
    }

    /**
     * Take n samples and labels from this dataset and return them in a new
     * dataset.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function take(int $n = 1) : self
    {
        if ($n < 0) {
            throw new InvalidArgumentException('Cannot take less than 0 samples.');
        }

        return $this->splice(0, $n);
    }

    /**
     * Leave n samples and labels on this dataset and return the rest in a new
     * dataset.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function leave(int $n = 1) : self
    {
        if ($n < 0) {
            throw new InvalidArgumentException('Cannot leave less than 0 samples.');
        }

        return $this->splice($n, $this->numRows());
    }

    /**
     * Prepend this dataset with another dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function prepend(Dataset $dataset) : Dataset
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Can only merge with a labeled'
                . 'dataset.');
        }

        $samples = array_merge($dataset->samples(), $this->samples);
        $labels = array_merge($dataset->labels(), $this->labels);

        return self::quick($samples, $labels);
    }

    /**
     * Append this dataset with another dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function append(Dataset $dataset) : Dataset
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Can only merge with a labeled'
                . 'dataset.');
        }

        $samples = array_merge($this->samples, $dataset->samples());
        $labels = array_merge($this->labels, $dataset->labels());

        return self::quick($samples, $labels);
    }

    /**
     * Remove a size n chunk of the dataset starting at offset and return it in
     * a new dataset.
     *
     * @param  int  $offset
     * @param  int  $n
     * @return self
     */
    public function splice(int $offset, int $n) : self
    {
        $samples = array_splice($this->samples, $offset, $n);
        $labels = array_splice($this->labels, $offset, $n);

        return self::quick($samples, $labels);
    }

    /**
     * Randomize the dataset in place and return self for chaining.
     *
     * @return self
     */
    public function randomize() : self
    {
        $order = range(0, $this->numRows() - 1);

        shuffle($order);

        array_multisort($order, $this->samples, $this->labels);

        return $this;
    }

    /**
     * Run a filter over the dataset using the values of a given column.
     *
     * @param  int  $index
     * @param  callable  $fn
     * @return self
     */
    public function filterByColumn(int $index, callable $fn) : self
    {
        $samples = $labels = [];

        foreach ($this->samples as $i => $sample) {
            if ($fn($sample[$index])) {
                $samples[] = $sample;
                $labels[] = $this->labels[$i];
            }
        }

        return self::quick($samples, $labels);
    }

    /**
     * Run a filter over the dataset using the labels for comparison.
     *
     * @param  callable  $fn
     * @return self
     */
    public function filterByLabel(callable $fn) : self
    {
        $samples = $labels = [];

        foreach ($this->labels as $i => $label) {
            if ($fn($label)) {
                $samples[] = $this->samples[$i];
                $labels[] = $label;
            }
        }

        return self::quick($samples, $labels);
    }

    /**
     * Sort the dataset in place by a column in the sample matrix.
     *
     * @param  int  $index
     * @param  bool  $descending
     * @return self
     */
    public function sortByColumn(int $index, bool $descending = false)
    {
        $order = $this->column($index);

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
     * @param  bool  $descending
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function sortByLabel(bool $descending = false) : Dataset
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
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0 or $ratio >= 1) {
            throw new InvalidArgumentException('Split ratio must be strictly'
            . " between 0 and 1, $ratio given.");
        }

        $n = (int) ($ratio * $this->numRows());

        $leftSamples = array_slice($this->samples, 0, $n);
        $leftLabels = array_slice($this->labels, 0, $n);

        $rightSamples = array_slice($this->samples, $n);
        $rightLabels = array_slice($this->labels, $n);

        return [
            self::quick($leftSamples, $leftLabels),
            self::quick($rightSamples, $rightLabels),
        ];
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self[]
     */
    public function stratifiedSplit(float $ratio = 0.5) : array
    {
        if ($ratio <= 0. or $ratio >= 1.) {
            throw new InvalidArgumentException('Split ratio must be strictly'
            . " between 0 and 1, $ratio given.");
        }

        $leftSamples = $leftLabels = $rightSamples = $rightLabels = [];

        foreach ($this->_stratify() as $label => $stratum) {
            $n = (int) floor($ratio * count($stratum));

            $leftSamples = array_merge($leftSamples, array_splice($stratum, 0, $n));
            $leftLabels = array_merge($leftLabels, array_fill(0, $n, $label));

            $rightSamples = array_merge($rightSamples, $stratum);
            $rightLabels = array_merge($rightLabels, array_fill(0, count($stratum), $label));
        }

        return [
            self::quick($leftSamples, $leftLabels),
            self::quick($rightSamples, $rightLabels),
        ];
    }

    /**
     * Fold the dataset k - 1 times to form k equal size datasets.
     *
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return array
     */
    public function fold(int $k = 10) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot create less than 2'
                . " folds, $k given.");
        }

        $n = (int) floor($this->numRows() / $k);

        $folds = [];

        for ($i = 0; $i < $k; $i++) {
            $offset = $i * $n;

            $samples = array_slice($this->samples, $offset, $n);
            $labels = array_slice($this->labels, $offset, $n);

            $folds[] = self::quick($samples, $labels);
        }

        return $folds;
    }

    /**
     * Fold the dataset into k equal sized stratified datasets.
     *
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return array
     */
    public function stratifiedFold(int $k = 10) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot create less than 2'
                . " folds, $k given.");
        }

        $folds = [];

        for ($i = 0; $i < $k; $i++) {
            $samples = $labels = [];

            foreach ($this->_stratify() as $label => $stratum) {
                $n = (int) floor(count($stratum) / $k);

                $samples = array_merge($samples, array_slice($stratum, $i * $n, $n));
                $labels = array_merge($labels, array_fill(0, $n, $label));
            }

            $folds[] = self::quick($samples, $labels);
        }

        return $folds;
    }

    /**
     * Stratifying subroutine groups samples by label.
     *
     * @return array[]
     */
    protected function _stratify() : array
    {
        $strata = [];

        foreach ($this->labels as $index => $label) {
            $strata[$label][] = $this->samples[$index];
        }

        return $strata;
    }

    /**
     * Generate a collection of batches of size n from the dataset. If there are
     * not enough samples to fill an entire batch, then the dataset will contain
     * as many samples and labels as possible.
     *
     * @param  int  $n
     * @return array
     */
    public function batch(int $n = 50) : array
    {
        $sChunks = array_chunk($this->samples, $n);
        $lChunks = array_chunk($this->labels, $n);

        $batches = [];

        foreach ($sChunks as $i => $samples) {
            $batches[] = self::quick($samples, $lChunks[$i]);
        }

        return $batches;
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
     * @param  int  $index
     * @param  mixed  $value
     * @throws \InvalidArgumentException
     * @return array
     */
    public function partition(int $index, $value) : array
    {
        if (!is_string($value) and !is_numeric($value)) {
            throw new InvalidArgumentException('Value must be a string or'
                . ' numeric type, ' . gettype($value) . ' given.');
        }

        $leftSamples = $leftLabels = $rightSamples = $rightLabels = [];

        if ($this->columnType($index) === DataType::CATEGORICAL) {
            foreach ($this->samples as $i => $sample) {
                if ($sample[$index] === $value) {
                    $leftSamples[] = $sample;
                    $leftLabels[] = $this->labels[$i];
                } else {
                    $rightSamples[] = $sample;
                    $rightLabels[] = $this->labels[$i];
                }
            }
        } else {
            foreach ($this->samples as $i => $sample) {
                if ($sample[$index] < $value) {
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
     * Generate a random subset with replacement.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubsetWithReplacement(int $n) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . " less than 1 sample, $n given.");
        }

        $max = $this->numRows() - 1;

        $samples = $labels = [];

        for ($i = 0; $i < $n; $i++) {
            $index = rand(0, $max);

            $samples[] = $this->samples[$index];
            $labels[] = $this->labels[$index];
        }

        return self::quick($samples, $labels);
    }

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param  int  $n
     * @param  (int|float)[]  $weights
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomWeightedSubsetWithReplacement(int $n, array $weights) : self
    {
        if (count($weights) !== count($this->samples)) {
            throw new InvalidArgumentException('The number of weights must be'
                . ' equal to the number of samples in the dataset, '
                . count($this->samples) . ' needed, ' . count($weights)
                . ' given.');
        }

        $total = array_sum($weights);
        $max = (int) round($total * self::PHI);

        $samples = $labels = [];

        for ($i = 0; $i < $n; $i++) {
            $delta = rand(0, $max) / self::PHI;

            foreach ($weights as $row => $weight) {
                $delta -= $weight;

                if ($delta <= 0.) {
                    $samples[] = $this->samples[$row];
                    $labels[] = $this->labels[$row];

                    break 1;
                }
            }
        }

        return self::quick($samples, $labels);
    }

    /**
     * Specify data which should be serialized to JSON.
     *
     * @return mixed
     */
    public function jsonSerialize()
    {
        return [
            'samples' => $this->samples,
            'labels' => $this->labels,
        ];
    }
}
