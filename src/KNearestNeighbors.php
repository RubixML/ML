<?php

namespace Rubix\Engine;

use MathPHP\Statistics\Average;
use MathPHP\Statistics\Descriptive;
use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use Rubix\Engine\Graph\DistanceFunctions\DistanceFunction;
use SplPriorityQueue;

class KNearestNeighbors implements Classifier, Regression
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
     * @var \Rubix\Engine\Graph\DistanceFunctions\DistanceFunction
     */
    protected $distanceFunction;

    /**
     * The training samples.
     *
     * @var array
     */
    protected $samples;

    /**
     * The training outcomes.
     *
     * @var array
     */
    protected $outcomes;

    /**
     * @param  int  $k
     * @param  \Rubix\Engine\Graph\DistanceFunctions\DistanceFunction  $distanceFunction
     */
    public function __construct(int $k = 3, DistanceFunction $distanceFunction = null)
    {
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
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return void
     */
    public function train(SupervisedDataset $data) : void
    {
        list($this->samples, $this->outcomes) = $data->toArray();
    }

    /**
     * Compute the distances and locate the k nearest neighboring values.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array
    {
        $neighbors = $this->findNearestNeighbors($sample);

        if (is_string($neighbors[0])) {
            $outcomes = array_count_values($neighbors);

            $outcome = array_search(max($outcomes), $outcomes);

            $certainty = $outcomes[$outcome] / count($neighbors);
        } else {
            $outcome = Average::median($neighbors);

            $certainty = Descriptive::standardDeviation($neighbors);
        }

        return [
            'outcome' => $outcome,
            'certainty' => $certainty,
        ];
    }

    /**
     * Find the K closest neighbors to the given sample vector.
     *
     * @param  array  $sample
     * @return array
     */
    protected function findNearestNeighbors(array $sample) : array
    {
        $neighbors = new SplPriorityQueue();
        $k = $this->k;

        foreach ($this->samples as $i => $neighbor) {
            $distance = $this->distanceFunction->distance($sample, $neighbor);

            $neighbors->insert($this->outcomes[$i], 1 - $distance);
        }

        if ($k > count($neighbors)) {
            $k = count($neighbors);
        }

        return array_map(function ($i) use ($neighbors) {
            return $neighbors->extract();
        }, range(0, $k - 1));
    }
}
