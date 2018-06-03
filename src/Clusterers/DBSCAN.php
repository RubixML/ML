<?php

namespace Rubix\Engine\Clusterers;

use Rubix\Engine\Persistable;
use Rubix\Engine\Unsupervised;
use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Metrics\Distance\Distance;
use Rubix\Engine\Metrics\Distance\Euclidean;
use InvalidArgumentException;

class DBSCAN implements Unsupervised, Clusterer, Persistable
{
    const NOISE = 'noise';

    /**
     * The maximum distance between two points to be considered neighbors. The
     * smaller the value, the tighter the clusters will be.
     *
     * @var float
     */
    protected $radius;

    /**
     * The minimum number of points to from a dense region or cluster.
     *
     * @var int
     */
    protected $minDensity;

    /**
     * The distance function to use when computing the distances between points.
     *
     * @var \Rubix\Engine\Contracts\Distance
     */
    protected $distanceFunction;

    /**
     * @param  float  $radius
     * @param  int  $minDensity
     * @param  \Rubix\Engine\Contracts\Distance  $distanceFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $radius = 0.5, int $minDensity = 5, Distance $distanceFunction = null)
    {
        if ($radius < 0.0) {
            throw new InvalidArgumentException('Epsilon cannot be less than 0.');
        }

        if ($minDensity < 0) {
            throw new InvalidArgumentException('Minimum density must be a'
                . ' number greater than 0.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->radius = $radius;
        $this->minDensity = $minDensity;
        $this->distanceFunction = $distanceFunction;
    }

    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return array
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }
    }

    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $labels = [];
        $current = 0;

        foreach ($samples as $index => $sample) {
            if (isset($labels[$index])) {
                continue 1;
            }

            $neighbors = $this->groupNeighborsByDistance($sample, $samples);

            if (count($neighbors) < $this->minDensity) {
                $labels[$index] = self::NOISE;

                continue 1;
            }

            $labels[$index] = $current;

            $this->expand($samples, $neighbors, $labels, $current);

            $current++;
        }

        return $labels;
    }

    /**
     * Expand the cluster by computing the distance between a sample and each
     * member of the cluster.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @param  array  $neighbors
     * @param  array  $labels
     * @param  int  $current
     * @return void
     */
    protected function expand(Dataset $samples, array $neighbors, array &$labels, int $current) : void
    {
        while (!empty($neighbors)) {
            $index = array_pop($neighbors);

            if (isset($labels[$index])) {
                if ($labels[$index] === self::NOISE) {
                    $labels[$index] = $current;
                }

                continue 1;
            }

            $labels[$index] = $current;

            $seeds = $this->groupNeighborsByDistance($samples->row($index),
                $samples);

            if (count($seeds) >= $this->minDensity) {
                $neighbors = array_unique(array_merge($neighbors, $seeds));
            }
        }
    }

    /**
     * Group the samples into a region defined by their distance from a given
     * centroid.
     *
     * @param  array  $neighbor
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    protected function groupNeighborsByDistance(array $neighbor, Dataset $samples) : array
    {
        $neighbors = [];

        foreach ($samples as $index => $sample) {
            $distance = $this->distanceFunction->compute($neighbor, $sample);

            if ($distance <= $this->radius) {
                $neighbors[] = $index;
            }
        }

        return $neighbors;
    }
}
