<?php

namespace Rubix\ML\Datasets;

use InvalidArgumentException;
use RuntimeException;

class Labeled extends DataFrame implements Dataset
{
    /**
     * The labeled outcomes for each sample in the dataset.
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
     * Factory method to create a labeled dataset from an array of datasets.
     *
     * @param  array  $datasets
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function combine(array $datasets) : self
    {
        $samples = $labels = [];

        foreach ($datasets as $dataset) {
            if (!$dataset instanceof Labeled) {
                throw new InvalidArgumentException('Cannot merge non-Labeled'
                    . ' datasets, ' . get_class($dataset) . ' found.');
            }

            $samples = array_merge($samples, $dataset->samples());
            $labels = array_merge($labels, $dataset->labels());
        }

        return new self($samples, $labels);
    }

    /**
     * @param  array  $samples
     * @param  array  $labels
     * @param  mixed  $placeholder
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $samples, array $labels, $placeholder = '?')
    {
        if (count($samples) !== count($labels)) {
            throw new InvalidArgumentException('The ratio of samples to labels'
             . ' must be equal.');
        }

        foreach ($labels as &$label) {
            if (!is_string($label) and !is_numeric($label)) {
                throw new InvalidArgumentException('Label must be a string or'
                    . ' numeric type, ' . gettype($label) . ' found.');
            }

            if (is_string($label) and is_numeric($label)) {
                $label = (int) $label == $label
                    ? (int) $label
                    : (float) $label;
            }
        }

        $this->labels = array_values($labels);

        parent::__construct($samples, $placeholder);
    }

    /**
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
            throw new \InvalidArgumentException('Row at offset '
                . (string) $index . ' does not exist.');
        }

        return $this->labels[$index];
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
            array_slice($this->labels, 0, $n));
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
            array_slice($this->labels, -$n));
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
            array_splice($this->labels, $offset, $n));
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

            $strata[$label] = new self($stratum, $labels);
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

        return [
            new self(array_slice($this->samples, 0, $n),
                array_slice($this->labels, 0, $n)),
            new self(array_slice($this->samples, $n),
                array_slice($this->labels, $n)),
        ];
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
        if ($ratio <= 0 or $ratio >= 1) {
            throw new InvalidArgumentException('Split ratio must be strictly'
            . ' between 0 and 1.');
        }

        $left = $right = [[], []];

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

        $n = (int) (count($samples) / $k);

        $folds = [];

        for ($i = 0; $i < $k; $i++) {
            $folds[] = new self(array_splice($samples, 0, $n),
                array_splice($labels, 0, $n));
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
                $n = (int) (count($stratum) / $k);

                $samples = array_merge($samples, array_splice($stratum, 0, $n));
                $labels = array_merge($labels, array_fill(0, $n, $label));
            }

            $folds[] = new self($samples, $labels);
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
        $batches = [];

        $samples = $this->samples;
        $labels = $this->labels;

        while (!empty($samples)) {
            $batches[] = new self(array_splice($samples, 0, $n),
                array_splice($labels, 0, $n));
        }

        return $batches;
    }

    /**
     * Generate a random subset.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubset(int $n = 1) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . ' less than 1 sample.');
        }

        if ($n > $this->numRows()) {
            throw new InvalidArgumentException('Cannot generate a larger subset'
                . ' than the sample size.');
        }

        $indices = (array) array_rand($this->samples, $n);

        return new self(array_intersect_key($this->samples, $indices),
            array_intersect_key($this->labels, $indices));
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return self
     */
    public function randomSubsetWithReplacement(int $n = 1) : self
    {
        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate a subset of'
                . ' less than 1 sample.');
        }

        $subset = [[], []];

        for ($i = 0; $i < $n; $i++) {
            $index = array_rand($this->samples);

            $subset[0][] = $this->samples[$index];
            $subset[1][] = $this->labels[$index];
        }

        return new self(...$subset);
    }

    /**
     * Save the dataset to a serialized object file.
     *
     * @param  string  $path
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function save(string $path = '') : void
    {
        if (empty($path)) {
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
     * Stratifying routine groups samples by label.
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
