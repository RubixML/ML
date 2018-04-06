<?php

namespace Rubix\Engine;

use InvalidArgumentException;

class SupervisedDataset extends Dataset
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
     * The output type. i.e. categorical or continuous.
     *
     * @var int
     */
    protected $output;

    /**
     * Build a supervised dataset used for training and testing models. The assumption
     * is the that dataset contain 0 < n < ∞ feature columns where the last column is
     * always the labeled outcome.
     *
     * @param  iterable  $data
     * @return self
     */
    public static function build(iterable $data) : self
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
            throw new InvalidArgumentException('The number of samples must equal the number of outcomes.');
        }

        parent::__construct($samples);

        foreach ($outcomes as &$outcome) {
            if (!is_string($outcome) && !is_numeric($outcome)) {
                throw new InvalidArgumentException('Outcome must be a string or numeric type, ' . gettype($outcome) . ' found.');
            }

            if (is_string($outcome) && is_numeric($outcome)) {
                $outcome = $this->convertNumericString($outcome);
            }
        }

        $this->output = is_string($outcomes[0]) ? static::CATEGORICAL : static::CONTINUOUS;

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
     * @return int
     */
    public function output() : int
    {
        return $this->output;
    }

    /**
     * The set of all possible labeled outcomes.
     *
     * @return array
     */
    public function labels() : array
    {
        return array_unique($this->outcomes);
    }

    /**
     * Randomize the dataset.
     *
     * @return self
     */
    public function randomize() : self
    {
        $order = range(0, count($this->outcomes) - 1);

        shuffle($order);

        array_multisort($order, $this->samples, $this->outcomes);

        return $this;
    }

    /**
     * Take n samples and outcomes from this dataset and return them in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function take(int $n = 1) : self
    {
        return new static(array_splice($this->samples, 0, $n), array_splice($this->outcomes, 0, $n));
    }

    /**
     * Leave n samples and outcomes on this dataset and return the rest in a new dataset.
     *
     * @param  int  $n
     * @return self
     */
    public function leave(int $n = 1) : self
    {
        return new static(array_splice($this->samples, $n), array_splice($this->outcomes, $n));
    }

    /**
     * Split the dataset into two stratified subsets with a given ratio of samples.
     *
     * @param  float  $ratio
     * @return array
     */
    public function split(float $ratio = 0.5) : array
    {
        if ($ratio <= 0.0 || $ratio >= 0.9) {
            throw new InvalidArgumentException('Split ratio must be a float value between 0.0 and 0.9.');
        }

        $strata = $this->stratify();

        $training = $testing = [0 => [], 1 => []];

        foreach ($strata[0] as $i => $stratum) {
            $testing[0] = array_merge($testing[0], array_splice($stratum, 0, round($ratio * count($stratum))));
            $testing[1] = array_merge($testing[1], array_splice($strata[1][$i], 0, round($ratio * count($strata[1][$i]))));

            $training[0] = array_merge($training[0], $stratum);
            $training[1] = array_merge($training[1], $strata[1][$i]);
        }

        return [
            new static(...$training),
            new static(...$testing),
        ];
    }

    /**
     * Generate a random subset with replacement.
     *
     * @param  float  $ratio
     * @return self
     */
    public function generateRandomSubset(float $ratio = 0.1) : self
    {
        $n = ceil($ratio * $this->rows());
        $max = $this->rows() - 1;
        $subset = [];

        foreach (range(1, $n) as $i) {
            $index = random_int(0, $max);

            $subset[0][] = $this->samples[$index];
            $subset[1][] = $this->outcomes[$index];
        }

        return new static(...$subset);
    }

    /**
     * Group samples by outcome and return an array of strata.
     *
     * @return array
     */
    public function stratify() : array
    {
        $strata = [];

        foreach ($this->outcomes as $i => $outcome) {
            $strata[0][$outcome][] = $this->samples[$i];
            $strata[1][$outcome][] = $outcome;
        }

        return $strata;
    }

    /**
     * @return array
     */
    public function toArray() : array
    {
        return [
            $this->samples,
            $this->outcomes,
        ];
    }
}
