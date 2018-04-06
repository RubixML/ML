<?php

namespace Rubix\Engine;

use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use Rubix\Engine\Graph\DistanceFunctions\DistanceFunction;
use SplPriorityQueue;

class KNearestNeighbors implements Classifier, Regression
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

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
     * The output type. i.e. categorical or continuous.
     *
     * @var int
     */
    protected $output;

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
        $this->output = 0;
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

        $this->output = $data->output();
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

        $n = count($neighbors);

        if ($this->output === self::CATEGORICAL) {
            $outcomes = array_count_values($neighbors);

            $outcome = array_search(max($outcomes), $outcomes);

            $certainty = $outcomes[$outcome] / $n;
        } else {
            $outcome = array_sum($neighbors) / $n;

            $certainty = sqrt(array_reduce($neighbors, function ($carry, $value) use ($outcome) {
                return $carry += ($value - $outcome) ** 2;
            }, 0) / $n);
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
