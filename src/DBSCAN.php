<?php

namespace Rubix\Engine;

use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use Rubix\Engine\Graph\DistanceFunctions\DistanceFunction;
use InvalidArgumentException;

class DBSCAN implements Clusterer
{
    const NOISE = null;

    /**
     * The minimum distance between two points.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * The minimum number of points to from a dense region.
     *
     * @var int
     */
    protected $minDensity;

    /**
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\Engine\Graph\DistanceFunctions\DistanceFunction
     */
    protected $distanceFunction;

    /**
     * The learned labels of the training data.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * @param  float  $epsilon
     * @param  int  $minPoints
     * @param  \Rubix\Engine\Graph\DistanceFunctions\DistanceFunction  $distanceFunction
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
     * @param  \Rubix\Engine\Dataset  $data
     * @throws \InvalidArgumentException
     * @return array
     */
    public function cluster(Dataset $data) : array
    {
        if (in_array(self::CATEGORICAL, $data->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        $this->labels = [];
        $n = 0;

        foreach ($data as $id => $sample) {
            if (isset($this->labels[$id])) {
                continue 1;
            }

            $neighbors = $this->groupNeighborsByDistance($sample, $data->samples());

            if (count($neighbors) < $this->minDensity) {
                $this->labels[$id] = self::NOISE;

                continue 1;
            }

            $this->labels[$id] = $n;

            $this->expand($data->samples(), $neighbors, $n);

            $n++;
        }

        $clusters = array_fill(0, $n, []);

        foreach ($data as $id => $sample) {
            if ($this->labels[$id] !== self::NOISE) {
                $clusters[$this->labels[$id]][] = $sample;
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
     * @param  int  $n
     * @return void
     */
    protected function expand(array $samples, array $neighbors, int $n) : void
    {
        while (!empty($neighbors)) {
            $id = array_pop($neighbors);

            if (isset($this->labels[$id])) {
                if ($this->labels[$id] === self::NOISE) {
                    $this->labels[$id] = $n;
                }

                continue 1;
            }

            $this->labels[$id] = $n;

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
            $distance = $this->distanceFunction->distance($neighbor, $sample);

            if ($distance <= $this->epsilon) {
                $neighbors[] = $id;
            }
        }

        return $neighbors;
    }
}
