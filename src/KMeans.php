<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Unsupervised;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\Metrics\DistanceFunctions\Euclidean;
use Rubix\Engine\Metrics\DistanceFunctions\DistanceFunction;
use InvalidArgumentException;
use RuntimeException;

class KMeans implements Clusterer, Persistable
{
    /**
     * The target number of clusters.
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
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $maxIterations;

    /**
     * @param  int  $k
     * @param  \Rubix\Engine\Contracts\DistanceFunction  $distanceFunction
     * @param  int  $maxIterations
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k, DistanceFunction $distanceFunction = null, int $maxIterations = PHP_INT_MAX)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be an integer greater than 1.');
        }

        if ($maxIterations < 1) {
            throw new InvalidArgumentException('Max interations must be greater than 1.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->k = $k;
        $this->maxIterations = $maxIterations;
        $this->distanceFunction = $distanceFunction;
    }

    /**
     * Cluster a given dataset based on the computed means of the centroids.
     *
     * @param  \Rubix\Engine\Datasets\Unsupervised  $dataset
     * @return array
     */
    public function cluster(Unsupervised $dataset) : array
    {
        $centroids = $this->findKCentroids($dataset->samples());

        $clusters = [];

        foreach ($dataset as $sample) {
            $label = $this->label($sample, $centroids);

            $clusters[$label][] = $sample;
        }

        return $clusters;
    }

    /**
     * Pick K random points and assign them as centroids. Compute the coordinates
     * of the centroids by clustering the points based on each sample's distance
     * from one of the k centroids, then recompute the centroid coordinate as the
     * mean of the new cluster's population.
     *
     * @param  array  $points
     * @throws \RuntimeException
     * @return array
     */
    protected function findKCentroids(array $points) : array
    {
        if (count($points) < $this->k) {
            throw new RuntimeException('The number of sample points cannot be less than K.');
        }

        $labels = array_fill(0, count($points), null);
        $sizes = array_fill(0, $this->k, 0);
        $changed = true;

        shuffle($points);

        $centroids = array_splice($points, 0, $this->k);

        for ($i = 0; $i < $this->maxIterations; $i++) {
            foreach ($points as $id => $sample) {
                $label = $this->label($sample, $centroids);

                if ($label !== $labels[$id]) {
                    $labels[$id] = $label;

                    $sizes[$label]++;
                } else {
                    $changed = false;
                }

                $n = $sizes[$label];

                foreach ($centroids[$label] as $column => &$mean) {
                    $mean = ($mean * ($n - 1) + $sample[$column]) / $n;
                }
            }

            if ($changed === false) {
                break 1;
            }
        }

        return $centroids;
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param  array  $sample
     * @param  array  $centroids
     * @return int
     */
    protected function label(array $sample, array $centroids) : int
    {
        $best = ['distance' => INF, 'label' => null];

        foreach ($centroids as $label => $centroid) {
            $distance = $this->distanceFunction->compute($sample, $centroid);

            if ($distance < $best['distance']) {
                $best = [
                    'distance' => $distance,
                    'label' => $label,
                ];
            }
        }

        return $best['label'];
    }
}
