<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Structures\DataFrame;
use InvalidArgumentException;

/**
 * DBSCAN
 *
 * Density-Based Spatial Clustering of Applications with Noise is a clustering
 * algorithm able to find non-linearly separable and arbitrarily-shaped
 * clusters. In addition, DBSCAN also has the ability to mark outliers as noise
 * and thus can be used as a quasi Anomaly Detector as well.
 *
 * References:
 * [1] M. Ester et al. (1996). A Densty-Based Algorithmfor Discovering Clusters.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DBSCAN implements Estimator, Persistable
{
    const NOISE = -1;

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
     * The distance kernel to use when computing the distances between points.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * @param  float  $radius
     * @param  int  $minDensity
     * @param  \Rubix\ML\Kernels\Distance\Distance  $kernel
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $radius = 0.5, int $minDensity = 5, Distance $kernel = null)
    {
        if ($radius < 0.) {
            throw new InvalidArgumentException('Radius cannot be less than 0.');
        }

        if ($minDensity < 0) {
            throw new InvalidArgumentException('Minimum density must be a'
                . ' number greater than 0.');
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->radius = $radius;
        $this->minDensity = $minDensity;
        $this->kernel = $kernel;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLUSTERER;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $samples = $dataset->samples();

        $predictions = [];
        $cluster = 0;

        foreach ($samples as $i => $sample) {
            if (isset($predictions[$i])) {
                continue 1;
            }

            $neighbors = $this->groupNeighbors($sample, $samples);

            if (count($neighbors) < $this->minDensity) {
                $predictions[$i] = self::NOISE;

                continue 1;
            }

            $predictions[$i] = $cluster;

            while (!empty($neighbors)) {
                $index = array_pop($neighbors);

                if (isset($predictions[$index])) {
                    if ($predictions[$index] === self::NOISE) {
                        $predictions[$index] = $cluster;
                    }

                    continue 1;
                }

                $predictions[$index] = $cluster;

                $centroid = $dataset->row($index);

                $seeds = $this->groupNeighbors($centroid, $samples);

                if (count($seeds) >= $this->minDensity) {
                    $neighbors = array_unique(array_merge($neighbors, $seeds));
                }
            }

            $cluster++;
        }

        return $predictions;
    }

    /**
     * Group the samples that are within a given radius of the centroid into a
     * neighborhood.
     *
     * @param  array  $centroid
     * @param  array  $samples
     * @return array
     */
    protected function groupNeighbors(array $centroid, array $samples) : array
    {
        $neighborhood = [];

        foreach ($samples as $i => $sample) {
            $distance = $this->kernel->compute($sample, $centroid);

            if ($distance <= $this->radius) {
                $neighborhood[] = $i;
            }
        }

        return $neighborhood;
    }
}
