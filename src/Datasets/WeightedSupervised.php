<?php

namespace Rubix\Engine\Datasets;

use InvalidArgumentException;

class WeightedSupervised extends Supervised
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
        if (!isset($weights) && !empty($samples)) {
            $weights = array_fill(0, count($samples), 1 / count($samples));
        }

        if (count($samples) !== count($weights)) {
            throw new InvalidArgumentException('The ratio of samples to weights must be equal.');
        }

        foreach ($weights as $row => $weight) {
            $this->setWeight($row, (float) $weight);
        }

        parent::__construct($samples, $outcomes);
    }

    /**
     * @return array
     */
    public function weights() : array
    {
        return $this->weights;
    }

    /**
     * Return the weight of a particular sample given by row offset.
     *
     * @param  int  $row
     * @return float
     */
    public function weight(int $row) : float
    {
        if (!isset($this->weights[$row])) {
            throw new RuntimeException('Invalid row offset.');
        }

        return $this->weights[$row];
    }

    /**
     * Return the sum of all the weights in the dataset.
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
     * @param  float  $weight
     * @throws \InvalidArgumentException
     * @return void
     */
    public function setWeight(int $row, float $weight) : self
    {
        if ($weight < 0) {
            throw new InvalidArgumentException('Weight value must be positive.');
        }

        $this->weights[$row] = $weight;

        return $this;
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

        array_multisort($order, $this->samples, $this->outcomes, $this->weights);

        return $this;
    }

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomSubsetWithReplacement(float $ratio = 0.1) : Dataset
    {
        if ($ratio < 0.0 || $ratio > 1.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value between 0 and 1.');
        }

        $n = round($ratio * $this->rows());
        $samples = $outcomes = $weights = [];

        for ($i = 0; $i < $n; $i++) {
            $row = $this->chooseRandomWeightedSample();

            $samples[] = $this->samples[$row];
            $outcomes[] = $this->outcomes[$row];
            $weights[] = $this->weights[$row];
        }

        return new self($samples, $outcomes, $weights);
    }

    /**
     * Chose a random weighted sample and return its row index.
     *
     * @return int
     */
    public function chooseRandomWeightedSample() : int
    {
        $total = $this->totalWeight();
        $scale = pow(10, 8);

        $random = random_int(0, $total * $scale) / $scale;

        for ($row = 0; $row < $this->rows(); $row++) {
            $random -= $this->weights[$row];

            if ($random < 0) {
                break 1;
            }
        }

        return $row;
    }
}
