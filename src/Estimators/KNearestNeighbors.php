<?php

namespace Rubix\Engine\Estimators;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Metrics\Distance\Distance;
use Rubix\Engine\Metrics\Distance\Euclidean;
use Rubix\Engine\Estimators\Predictions\Probabalistic;
use InvalidArgumentException;
use SplPriorityQueue;

class KNearestNeighbors implements Classifier
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
     * @var \Rubix\Engine\Metrics\Distance\Distance
     */
    protected $distanceFunction;

    /**
     * The memoized coordinate vectors of the training data.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * The memoized labels.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * @param  int  $k
     * @param  \Rubix\Engine\Metrics\Distance\Distance  $distanceFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 3, Distance $distanceFunction = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . ' to make a prediction.');
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
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous samples.');
        }

        list($this->samples, $this->labels) = $dataset->all();
    }

    /**
     * Compute the distances and locate the k nearest neighboring values.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($samples as $sample) {
            $neighbors = $this->findNearestNeighbors($sample);

            $counts = array_count_values($neighbors);

            $outcome = array_search(max($counts), $counts);

            $probability = $counts[$outcome] / count($neighbors);

            $predictions[] = new Probabalistic($outcome, $probability);
        }

        return $predictions;
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

        foreach ($this->samples as $index => $neighbor) {
            $distance = $this->distanceFunction->compute($sample, $neighbor);

            $computed->insert($this->labels[$index], -$distance);
        }

        $n = (count($this->samples) >= $this->k
            ? $this->k : count($this->samples));

        for ($i = 0; $i < $n; $i++) {
            $neighbors[] = $computed->extract();
        }

        return $neighbors;
    }
}
