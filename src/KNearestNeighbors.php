<?php

namespace Rubix\Engine;

use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use Rubix\Engine\Graph\DistanceFunctions\DistanceFunction;
use InvalidArgumentException;
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
     * The coordinate vectors of the training data.
     *
     * @var array
     */
    protected $coordinates = [
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
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 3, DistanceFunction $distanceFunction = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K cannot be less than 1.');
        }

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
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $data) : void
    {
        if (!$data instanceof SupervisedDataset) {
            throw new InvalidArgumentException('This estimator requires a supervised dataset.');
        }

        if (in_array(self::CATEGORICAL, $data->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        list($this->coordinates, $this->outcomes) = $data->toArray();

        $this->output = $data->output();
    }

    /**
     * Compute the distances and locate the k nearest neighboring values.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $outcomes = $this->findNearestNeighbors($sample);

        if ($this->output === self::CATEGORICAL) {
            $counts = array_count_values($outcomes);

            $outcome = array_search(max($counts), $counts);

            $certainty = $counts[$outcome] / count($outcomes);

            return new Prediction($outcome, [
                'certainty' => $certainty,
            ]);
        } else {
            $mean = Average::mean($outcomes);

            $variance = array_reduce($outcomes, function ($carry, $outcome) use ($mean) {
                return $carry += ($outcome - $mean) ** 2;
            }, 0.0) / count($outcomes);

            return new Prediction($mean, [
                'variance' => $variance,
            ]);
        }

        return new Prediction($outcome, $certainty);
    }

    /**
     * Find the K nearest neighbors to the given sample vector.
     *
     * @param  array  $sample
     * @return array
     */
    protected function findNearestNeighbors(array $sample) : array
    {
        $neighbors = new SplPriorityQueue();
        $k = $this->k;

        foreach ($this->coordinates as $i => $neighbor) {
            $distance = $this->distanceFunction->compute($sample, $neighbor);

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
