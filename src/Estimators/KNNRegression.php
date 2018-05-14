<?php

namespace Rubix\Engine\Estimators;

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Predictions\Prediction;
use Rubix\Engine\Metrics\DistanceFunctions\Euclidean;
use Rubix\Engine\Metrics\DistanceFunctions\DistanceFunction;
use InvalidArgumentException;
use SplPriorityQueue;

class KNNRegression implements Regressor
{
    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected $k;

    /**
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\Engine\Contracts\DistanceFunction
     */
    protected $distanceFunction;

    /**
     * The coordinate vectors of the training data.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * The training outcomes.
     *
     * @var array
     */
    protected $outcomes = [
        //
    ];

    /**
     * @param  int  $k
     * @param  \Rubix\Engine\Contracts\DistanceFunction  $distanceFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 3, DistanceFunction $distanceFunction = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required to make a prediction.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->k = $k;
        $this->distanceFunction = $distanceFunction;
    }

    /**
     * Store the sample and outcome arrays. No other work to be done as this is
     * a lazy learning algorithm.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        $this->samples = $dataset->samples();
        $this->outcomes = $dataset->outcomes();
    }

    /**
     * Compute the distances and locate the k nearest neighboring values.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Estimaotors\Predictions\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $outcomes = $this->findNearestNeighbors($sample);

        $mean = Average::mean($outcomes);

        $variance = array_reduce($outcomes, function ($carry, $outcome) use ($mean) {
            return $carry += ($outcome - $mean) ** 2;
        }, 0.0) / count($outcomes);

        return new Prediction($mean);
    }

    /**
     * Find the K nearest neighbors to the given sample vector.
     *
     * @param  array  $sample
     * @return array
     */
    protected function findNearestNeighbors(array $sample) : array
    {
        $computed = new SplPriorityQueue();
        $neighbors = [];

        foreach ($this->samples as $row => $neighbor) {
            $distance = $this->distanceFunction->compute($sample, $neighbor);

            $computed->insert($this->outcomes[$row], -$distance);
        }

        $n = (count($this->samples) >= $this->k ? $this->k : count($this->samples));

        for ($i = 0; $i < $n; $i++) {
            $neighbors[] = $computed->extract();
        }

        return $neighbors;
    }
}
