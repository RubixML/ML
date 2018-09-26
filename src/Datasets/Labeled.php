<?php

namespace Rubix\ML\Datasets;

use Rubix\ML\Other\Structures\DataFrame;
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
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * Restore a labeled dataset from a serialized object file.
     *
     * @param  string  $path
     * @return self
     */
    public static function restore(string $path) : self
    {
        if (!file_exists($path) or !is_readable($path)) {
            throw new RuntimeException('File ' . basename($path) . ' cannot be'
                . ' opened. Check path and permissions.');
        }

        $dataset = unserialize(file_get_contents($path) ?: '');

        if (!$dataset instanceof Labeled) {
            throw new RuntimeException('Dataset could not be reconstituted.');
        }

        return $dataset;
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
            throw new InvalidArgumentException('The ratio of samples to labels'
             . ' must be equal.');
        }

        if ($validate === true) {
            $labels = array_values($labels);

            foreach ($labels as &$label) {
                if (!is_string($label) and !is_numeric($label)) {
                    throw new InvalidArgumentException('Label must be a string'
                        . ' or numeric type, ' . gettype($label) . ' found.');
                }

                if (is_string($label) and is_numeric($label)) {
                    $label = (int) $label == $label
                        ? (int) $label
                        : (float) $label;
                }
            }
        }

        $this->labels = $labels;

        parent::__construct($samples, $validate);
    }

    /**
     * Return all of the labels.
     *
     * @return array
     */
    public function labels() : array
    {
        return $this->labels;
    }

    /**
     * Return a label given by row index.
     *
     * @param  int  $index
     * @throws \InvalidArgumentException
     * @return mixed
     */
    public function label(int $index)
    {
        if (!isset($this->labels[$index])) {
            throw new InvalidArgumentException('Row at offset '
                . (string) $index . ' does not exist.');
        }

        return $this->labels[$index];
    }

    /**
     * Return the integer encoded data type of the label.
     *
     * @return int
     */
    public function labelType() : int
    {
        return is_string($this->labels[0])
            ? DataFrame::CATEGORICAL
            : DataFrame::CONTINUOUS;
    }

    /**
     * The set of all possible labeled outcomes.
     *
     * @return array
     */
    public function possibleOutcomes() : array
    {
        return array_values(array_unique($this->labels));
    }

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function head(int $n = 10) : self
    {
        return new self(array_slice($this->samples, 0, $n),
            array_slice($this->labels, 0, $n), false);
    }

    /**
     * Return a dataset containing only the last n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function tail(int $n = 10) : self
    {
        return new self(array_slice($this->samples, -$n),
            array_slice($this->labels, -$n), false);
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
     * Merge this dataset with another dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function merge(Dataset $dataset) : Dataset
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Can only merge with a labeled'
                . 'dataset.');
        }

        $samples = array_merge($this->samples, $dataset->samples());
        $labels = array_merge($this->labels, $dataset->labels());

        return new self($samples, $labels, false);
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
        return new self(array_splice($this->samples, $offset, $n),
            array_splice($this->labels, $offset, $n), false);
    }

    /**
     * Randomize the dataset.
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
            if ($fn($sample[$index]) === true) {
                $samples[] = $sample;
                $labels[] = $this->labels[$i];
            }
        }

        return new self($samples, $labels);
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
            if ($fn($label) === true) {
                $samples[] = $this->samples[$i];
                $labels[] = $label;
            }
        }

        return new self($samples, $labels);
    }

    /**
     * Sort the dataset by a column in the sample matrix.
     *
     * @param  int  $index
     * @param  bool  $descending
     * @return self
     */
    public function sortByColumn(int $index, bool $descending = false)
    {
        $order = $this->column($index);

        array_multisort($order, $this->samples, $this->labels,
            $descending ? SORT_DESC : SORT_ASC);

        return $this;
    }

    /**
     * Sort the dataset by its labels.
     *
     * @param  bool  $descending
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function sortByLabel(bool $descending = false) : Dataset
    {
        array_multisort($this->labels, $this->samples,
            $descending ? SORT_DESC : SORT_ASC);

        return $this;
    }

    /**
     * Group samples by label and return an array of stratified datasets. i.e.
     * n datasets consisting of samples with the same label where n is equal to
     * the number of unique labels.
     *
     * @return array
     */
    public function stratify() : array
    {
        $strata = [];

        foreach ($this->_stratify() as $label => $stratum) {
            $labels = array_fill(0, count($stratum), $label);

            $strata[$label] = new self($stratum, $labels, false);
        }

        return $strata;
    }

    /**
     * Split the dataset into two subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return array
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0 or $ratio >= 1) {
            throw new InvalidArgumentException('Split ratio must be strictly'
            . ' between 0 and 1.');
        }

        $n = (int) ($ratio * $this->numRows());

        $left = new self(array_slice($this->samples, 0, $n),
            array_slice($this->labels, 0, $n), false);
        $right = new self(array_slice($this->samples, $n),
            array_slice($this->labels, $n), false);

        return [$left, $right];
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return array
     */
    public function stratifiedSplit(float $ratio = 0.5) : array
    {
        if ($ratio <= 0. or $ratio >= 1.) {
            throw new InvalidArgumentException('Split ratio must be strictly'
            . ' between 0 and 1.');
        }

        $left = $right = [[], [], false];

        foreach ($this->_stratify() as $label => $stratum) {
            $n = (int) ($ratio * count($stratum));

            $left[0] = array_merge($left[0], array_splice($stratum, 0, $n));
            $left[1] = array_merge($left[1], array_fill(0, $n, $label));

            $right[0] = array_merge($right[0], $stratum);
            $right[1] = array_merge($right[1], array_fill(0, count($stratum), $label));
        }

        return [new self(...$left), new self(...$right)];
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
            throw new InvalidArgumentException('Cannot fold the dataset less than'
            . '1 time.');
        }

        $samples = $this->samples;
        $labels = $this->labels;

        $n = (int) floor(count($samples) / $k);

        $folds = [];

        for ($i = 0; $i < $k; $i++) {
            $folds[] = new self(array_splice($samples, 0, $n),
                array_splice($labels, 0, $n), false);
        }

        return $folds;
    }

    /**
     * Fold the dataset k - 1 times to form k equal size stratified datasets.
     *
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return array
     */
    public function stratifiedFold(int $k = 10) : array
    {
        if ($k < 2) {
            throw new InvalidArgumentException('Cannot fold the dataset less'
                . ' than 1 time.');
        }

        $folds = [];

        for ($i = 0; $i < $k; $i++) {
            $samples = $labels = [];

            foreach ($this->_stratify() as $label => $stratum) {
                $n = (int) floor(count($stratum) / $k);

                $samples = array_merge($samples, array_slice($stratum, $i * $n, $n));
                $labels = array_merge($labels, array_fill(0, $n, $label));
            }

            $folds[] = new self($samples, $labels, false);
        }

        return $folds;
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
            $batches[] = new self($samples, $lChunks[$i], false);
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
     * @return array
     */
    public function partition(int $index, $value) : array
    {
        $left = $right = [[], [], false];

        if ($this->columnType($index) === DataFrame::CATEGORICAL) {
            foreach ($this->samples as $i => $sample) {
                if ($sample[$index] === $value) {
                    $left[0][] = $sample;
                    $left[1][] = $this->labels[$i];
                } else {
                    $right[0][] = $sample;
                    $right[1][] = $this->labels[$i];
                }
            }
        } else {
            foreach ($this->samples as $i => $sample) {
                if ($sample[$index] < $value) {
                    $left[0][] = $sample;
                    $left[1][] = $this->labels[$i];
                } else {
                    $right[0][] = $sample;
                    $right[1][] = $this->labels[$i];
                }
            }
        }

        return [new self(...$left), new self(...$right)];
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
                . ' less than 1 sample.');
        }

        $max = $this->numRows() - 1;

        $samples = $labels = [];

        for ($i = 0; $i < $n; $i++) {
            $index = rand(0, $max);

            $samples[] = $this->samples[$index];
            $labels[] = $this->labels[$index];
        }

        return new self($samples, $labels, false);
    }

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param  int  $n
     * @param  array  $weights
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomWeightedSubsetWithReplacement(int $n, array $weights) : self
    {
        if (count($weights) !== count($this->samples)) {
            throw new InvalidArgumentException('The number of weights must be'
                . ' equal to the number of samples in the dataset.');
        }

        $total = array_sum($weights);
        $max = (int) round($total * self::PHI);

        $samples = $labels = [];

        for ($i = 0; $i < $n; $i++) {
            $delta = rand(0, $max) / self::PHI;

            foreach ($weights as $row => $weight) {
                $delta -= $weight;

                if ($delta < 0.) {
                    $samples[] = $this->samples[$row];
                    $labels[] = $this->labels[$row];

                    break 1;
                }
            }
        }

        return new self($samples, $labels, false);
    }

    /**
     * Save the dataset to a serialized object file.
     *
     * @param  string|null  $path
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function save(?string $path = null) : void
    {
        if (is_null($path)) {
            $path = (string) time() . '.dataset';
        }

        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not'
                . ' writable. Check path and permissions.');
        }

        $success = file_put_contents($path, serialize($this), LOCK_EX);

        if (!$success) {
            throw new RuntimeException('Failed to serialize object to storage.');
        }
    }

    /**
     * Stratifying subroutine groups samples by label.
     *
     * @return array
     */
    protected function _stratify() : array
    {
        $strata = [];

        foreach ($this->labels as $index => $label) {
            $strata[$label][] = $this->samples[$index];
        }

        return $strata;
    }
}
