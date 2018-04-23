<?php

namespace Rubix\Engine;

use InvalidArgumentException;

class WeightedDataset extends SupervisedDataset
{
    /**
     * The weight of each training sample in the dataset.
     *
     * @var array
     */
    protected $weights = [
        //
    ];

    /**
     * @param  array  $samples
     * @param  array  $outcomes
     * @param  array|null  $weights
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $samples, array $outcomes, ?array $weights = null)
    {
        if (!isset($weights)) {
            $weights = array_fill(0, count($samples), 1 / count($samples));
        }

        if (count($samples) !== count($weights)) {
            throw new InvalidArgumentException('The number of weights must equal the number of samples.');
        }

        foreach ($weights as $i => $weight) {
            $this->setWeight($i, $weight);
        }

        parent::__construct($samples, $outcomes);
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
     * Return the sum of all weights.
     *
     * @return mixed
     */
    public function totalWeight()
    {
        return array_sum($this->weights);
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
}
