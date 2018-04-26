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
     * The weight of each training sample in the dataset.
     *
     * @var array
     */
    protected $weights = [
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
     * @param  array|null  $weights
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $samples, array $outcomes, ?array $weights = null)
    {
        if (count($samples) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of samples must equal the number of outcomes.');
        }

        if (!isset($weights)) {
            $weights = array_fill(0, count($samples), 1 / count($samples));
        }

        if (count($samples) !== count($weights)) {
            throw new InvalidArgumentException('The number of weights must equal the number of samples.');
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

        foreach ($weights as $i => $weight) {
            $this->setWeight($i, $weight);
        }

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
    public function getOutcome(int $row)
    {
        return $this->outcomes[$row] ?? null;
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
     * Return the weight of a particular sample given by row offset.
     *
     * @param  int  $row
     * @return mixed
     */
    public function weight(int $row)
    {
        if (!isset($this->weights[$row])) {
            throw new RuntimeException('Inlvalid row offset.');
        }

        return $this->weights[$row];
    }

    /**
     * @return array
     */
    public function weights() : array
    {
        return $this->weights;
    }

    /**
     * Return the sum of all weights.
     *
     * @return mixed
     */
    public function totalWeight()
    {
        return array_sum($this->weights);
    }

    /**
     * Set the weight of a particular sample given by row offset.
     *
     * @param  int  $row
     * @param  mixed  $weights
     * @throws \InvalidArgumentException
     * @return void
     */
    public function setWeight(int $row, $weight) : void
    {
        if ((!is_int($weight) && !is_float($weight)) || $weight < 0) {
            throw new InvalidArgumentException('Weight must be an positive integer or float value.');
        }

        $this->weights[$row] = $weight;
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
     * Return a dataset containing only the first n samples.
     *
     * @param  int  $n
     * @return self
     */
    public function head(int $n = 10) : self
    {
        return new static(array_slice($this->samples, 0, $n), array_slice($this->outcomes, 0, $n));
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

        $testing = [
            array_splice($this->samples, 0, round($ratio * count($this->samples))),
            array_splice($this->outcomes, 0, round($ratio * count($this->outcomes))),
        ];

        return [
            new static($this->samples, $this->outcomes),
            new static($testing[0], $testing[1]),
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
            throw new InvalidArgumentException('Split ratio must be a float value between 0.0 and 1.0.');
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
     * Generate a random subset without replacement.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomSubset(float $ratio = 0.1) : self
    {
        if ($ratio <= 0.0 || $ratio >= 1.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value between 0 and 1.');
        }

        $n = round($ratio * $this->rows());

        $samples = $this->samples;
        $outcomes = $this->outcomes;

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
    public function generateRandomSubsetWithReplacement(float $ratio = 0.1) : self
    {
        if ($ratio <= 0.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value greater than 0.');
        }

        $max = $this->rows() - 1;
        $subset = [];

        foreach (range(1, round($ratio * $this->rows())) as $i) {
            $index = random_int(0, $max);

            $subset[0][] = $this->samples[$index];
            $subset[1][] = $this->outcomes[$index];
        }

        return new static(...$subset);
    }

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomWeightedSubsetWithReplacement(float $ratio = 0.1) : self
    {
        if ($ratio < 0.0 || $ratio > 1.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value between 0 and 1.');
        }

        $n = round($ratio * $this->rows());
        $samples = $outcomes = [];

        for ($i = 0; $i < $n; $i++) {
            $randomWeight = random_int(0, array_sum($this->weights) * 10000) / 10000;

            foreach ($this->samples as $j => $sample) {
                $randomWeight = $randomWeight - $this->weights[$j];

                if ($randomWeight < 0) {
                    $samples[] = $this->samples[$j];
                    $outcomes[] = $this->outcomes[$j];

                    break;
                }
            }
        }

        return new self($samples, $outcomes);
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

    /**
     * Return a new weighted dataset with uniform weights.
     *
     * @return \Rubix\Engine\WeightedDataset
     */
    public function toWeightedDataset() : WeightedDataset
    {
        return new WeightedDataset($this->samples, $this->outcomes);
    }
}
