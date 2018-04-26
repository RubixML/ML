<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Unsupervised;
use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use Rubix\Engine\Graph\DistanceFunctions\DistanceFunction;
use InvalidArgumentException;

class DBSCAN implements Clusterer
{
    const NOISE = null;

    /**
     * The maximum distance between two points to be considered neighbors. The
     * smaller the value, the tighter the clusters will be.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * The minimum number of points to from a dense region or cluster.
     *
     * @var int
     */
    protected $minDensity;

    /**
     * The distance function to use when computing the distances between points.
     *
     * @var \Rubix\Engine\Contracts\DistanceFunction
     */
    protected $distanceFunction;

    /**
     * @param  float  $epsilon
     * @param  int  $minDensity
     * @param  \Rubix\Engine\Contracts\DistanceFunction  $distanceFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $epsilon, int $minDensity = 5, DistanceFunction $distanceFunction = null)
    {
        if ($epsilon < 0.0) {
            throw new InvalidArgumentException('Epsilon cannot be less than 0.');
        }

        if ($minDensity < 0) {
            throw new InvalidArgumentException('Minimum density must be a number greater than 0.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->epsilon = $epsilon;
        $this->minDensity = $minDensity;
        $this->distanceFunction = $distanceFunction;
    }

    /**
     * @param  \Rubix\Engine\Datasets\Unsupervised  $dataset
     * @throws \InvalidArgumentException
     * @return array
     */
    public function cluster(Unsupervised $dataset) : array
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        $labels = [];
        $current = 0;

        foreach ($dataset as $id => $sample) {
            if (isset($labels[$id])) {
                continue 1;
            }

            $neighbors = $this->groupNeighborsByDistance($sample, $dataset->samples());

            if (count($neighbors) < $this->minDensity) {
                $labels[$id] = self::NOISE;

                continue 1;
            }

            $labels[$id] = $current;

            $this->expand($dataset->samples(), $neighbors, $labels, $current);

            $current++;
        }

        $clusters = array_fill(0, $current, []);

        foreach ($dataset as $id => $sample) {
            if ($labels[$id] !== self::NOISE) {
                $clusters[$labels[$id]][] = $sample;
            }
        }

        return $clusters;
    }

    /**
     * Expand the cluster by computing the distance between a sample and each
     * member of the cluster.
     *
     * @param  array  $samples
     * @param  array  $neighbors
     * @param  array  $labels
     * @param  int  $label
     * @return void
     */
    protected function expand(array $samples, array $neighbors, array &$labels, int $current) : void
    {
        while (!empty($neighbors)) {
            $id = array_pop($neighbors);

            if (isset($labels[$id])) {
                if ($labels[$id] === self::NOISE) {
                    $labels[$id] = $label;
                }

                continue 1;
            }

            $labels[$id] = $current;

            $seeds = $this->groupNeighborsByDistance($samples[$id], $samples);

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
     * @param  array  $samples
     * @return array
     */
    protected function groupNeighborsByDistance(array $neighbor, array $samples) : array
    {
        $neighbors = [];

        foreach ($samples as $id => $sample) {
            $distance = $this->distanceFunction->compute($neighbor, $sample);

            if ($distance <= $this->epsilon) {
                $neighbors[] = $id;
            }
        }

        return $neighbors;
    }
}
