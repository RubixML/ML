<?php

namespace Rubix\Engine\Datasets;

use InvalidArgumentException;

class Supervised extends Dataset
{
    /**
     * The labeled outcomes used for supervised training.
     *
     * @var array
     */
    protected $outcomes = [
        //
    ];

    /**
     * Build a supervised dataset used for training and testing models from an
     * iterator or array of feature vectors. The assumption is that the dataset
     * contains 0 < n < ∞ feature columns where the last column is always the
     * labeled outcome.
     *
     * @param  iterable  $data
     * @return self
     */
    public static function fromIterator(iterable $data) : self
    {
        $samples = $outcomes = [];

        foreach ($data as $row) {
            $outcomes[] = array_pop($row);
            $samples[] = array_values($row);
        }

        return new self($samples, $outcomes);
    }

    /**
     * @param  array  $samples
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $samples, array $outcomes)
    {
        if (count($samples) !== count($outcomes)) {
            throw new InvalidArgumentException('The ratio of samples to outcomes must be equal.');
        }

        foreach ($outcomes as &$outcome) {
            if (!is_string($outcome) && !is_numeric($outcome)) {
                throw new InvalidArgumentException('Outcome must be a string or numeric type, ' . gettype($outcome) . ' found.');
            }

            if (is_string($outcome) && is_numeric($outcome)) {
                $outcome = $this->convertNumericString($outcome);
            }
        }

        parent::__construct($samples);

        $this->outcomes = $outcomes;
    }

    /**
     * @return array
     */
    public function outcomes() : array
    {
        return $this->outcomes;
    }

    /**
     * Return the outcome at the given row.
     *
     * @param  int  $row
     * @return mixed
     */
    public function outcome(int $row)
    {
        if (!isset($this->outcomes[$row])) {
            throw new RuntimeException('Invalid row offset.');
        }

        return $this->outcomes[$row];
    }

    /**
     * The set of all possible labeled outcomes.
     *
     * @return array
     */
    public function labels() : array
    {
        return array_values(array_unique($this->outcomes));
    }

    /**
     * The type of data of the outcomes. i.e. categorical or continuous.
     *
     * @return int
     */
    public function outcomeType() : int
    {
        return is_string(reset($this->outcomes)) ? self::CATEGORICAL : self::CONTINUOUS;
    }

    /**
     * Randomize the dataset.
     *
     * @return self
     */
    public function randomize() : Dataset
    {
        $order = range(0, $this->rows() - 1);

        shuffle($order);

        array_multisort($order, $this->samples, $this->outcomes);

        return $this;
    }

    /**
     * Return a dataset containing only the first n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function head(int $n = 10) : Dataset
    {
        return new self(array_slice($this->samples, 0, $n), array_slice($this->outcomes, 0, $n));
    }

    /**
     * Take n samples and outcomes from this dataset and return them in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function take(int $n = 1) : Dataset
    {
        return new self(array_splice($this->samples, 0, $n), array_splice($this->outcomes, 0, $n));
    }

    /**
     * Leave n samples and outcomes on this dataset and return the rest in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function leave(int $n = 1) : Dataset
    {
        return new self(array_splice($this->samples, $n), array_splice($this->outcomes, $n));
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
        if ($ratio <= 0.0 || $ratio >= 1.0) {
            throw new InvalidArgumentException('Split ratio must be a float value between 0.0 and 1.0.');
        }

        $n = round($ratio * $this->rows());

        return [
            new self(array_splice($this->samples, 0, $n), array_splice($this->outcomes, 0, $n)),
            new self($this->samples, $this->outcomes),
        ];
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @return array
     */
    public function stratifiedSplit(float $ratio = 0.5) : array
    {
        if ($ratio <= 0.0 || $ratio >= 1.0) {
            throw new InvalidArgumentException('Split ratio must be between 0.0 and 1.0.');
        }

        $strata = $this->stratify();

        $left = $right = [[], []];
        $totals = [];

        foreach ($strata[0] as $label => $stratum) {
            $totals[$label] = count($stratum);
        }

        foreach ($strata[0] as $label => $stratum) {
            $n = round($ratio * $totals[$label]);

            $left[0] = array_merge($left[0], array_splice($stratum, 0, $n));
            $left[1] = array_merge($left[1], array_splice($strata[1][$label], 0, $n));

            $right[0] = array_merge($right[0], $stratum);
            $right[1] = array_merge($right[1], $strata[1][$label]);
        }

        return [
            new self(...$left),
            new self(...$right),
        ];
    }

    /**
     * Fold the dataset k times to form k + 1 equal size datasets.
     *
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return array
     */
    public function fold(int $k = 2) : array
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Cannot fold the dataset less than 1 time.');
        }

        $n = round(count($this->samples) / ($k + 1));
        $subsets = [];

        for ($i = 0; $i < $k + 1; $i++) {
            $subsets[] = new self(array_splice($this->samples, 0, $n), array_splice($this->outcomes, 0, $n));
        }

        return $subsets;
    }

    /**
     * Fold the dataset k times to form k + 1 equal size stratified datasets.
     *
     * @param  int  $k
     * @return array
     */
    public function stratifiedFold(int $k = 2) : array
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Cannot fold the dataset less than 1 time.');
        }

        $strata = $this->stratify();

        $subsets = $totals = [];

        foreach ($strata[0] as $label => $stratum) {
            $totals[$label] = count($stratum);
        }

        for ($i = 0; $i < $k + 1; $i++) {
            $samples = $outcomes = [];

            foreach ($strata[0] as $label => $stratum) {
                $n = $totals[$label] / ($k + 1);

                $samples = array_merge($samples, array_splice($stratum, 0, $n));
                $outcomes = array_merge($outcomes, array_splice($strata[1][$label], 0, $n));
            }

            $subsets[] = new self($samples, $outcomes);
        }

        return $subsets;
    }

    /**
     * Generate a random subset without replacement.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomSubset(float $ratio = 0.1) : Dataset
    {
        if ($ratio <= 0.0 || $ratio >= 1.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value between 0 and 1.');
        }

        $n = round($ratio * $this->rows());

        list($samples, $outcomes) = [$this->samples, $this->outcomes];

        $order = range(0, count($outcomes) - 1);

        shuffle($order);

        array_multisort($order, $samples, $outcomes);

        return new self(array_slice($samples, 0, $n), array_slice($outcomes, 0, $n));
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomSubsetWithReplacement(float $ratio = 0.1) : Dataset
    {
        if ($ratio <= 0.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value greater than 0.');
        }

        $n = round($ratio * $this->rows());
        $max = $this->rows() - 1;
        $subset = [[], []];

        for ($i = 0; $i < $n; $i++) {
            $row = random_int(0, $max);

            $subset[0][] = $this->samples[$row];
            $subset[1][] = $this->outcomes[$row];
        }

        return new self(...$subset);
    }

    /**
     * Group samples by outcome and return an array of strata.
     *
     * @return array
     */
    public function stratify() : array
    {
        $strata = [[], []];

        foreach ($this->outcomes as $i => $outcome) {
            $strata[0][$outcome][] = $this->samples[$i];
            $strata[1][$outcome][] = $outcome;
        }

        return $strata;
    }

    /**
     * Return an array of all the samples and outcomes with the last column of each
     * row being the labeled outcome of the supervised dataset.
     *
     * @return array
     */
    public function all() : array
    {
        return array_map(function ($sample, $outcome) {
            return array_merge($sample, (array) $outcome);
        }, $this->samples, $this->outcomes);
    }
}
