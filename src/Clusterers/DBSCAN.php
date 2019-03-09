<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\Graph\BallTree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;

/**
 * DBSCAN
 *
 * Density-Based Spatial Clustering of Applications with Noise is a clustering
 * algorithm able to find non-linearly separable and arbitrarily-shaped clusters.
 * In addition, DBSCAN also has the ability to mark outliers as *noise* and thus
 * can be used as a quasi Anomaly Detector.
 *
 * > **Note**: Noise samples are assigned the cluster number *-1*.
 *
 * References:
 * [1] M. Ester et al. (1996). A Densty-Based Algorithm for Discovering Clusters.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DBSCAN implements Estimator
{
    public const START_CLUSTER = 0;
    
    public const NOISE = -1;

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
     * The maximum number of samples that each ball node can contain.
     *
     * @var int
     */
    protected $maxLeafSize;

    /**
     * @param float $radius
     * @param int $minDensity
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param int $maxLeafSize
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $radius = 0.5,
        int $minDensity = 5,
        ?Distance $kernel = null,
        int $maxLeafSize = 30
    ) {
        if ($radius <= 0.) {
            throw new InvalidArgumentException('Cluster radius must be'
                . " greater than 0, $radius given.");
        }

        if ($minDensity <= 0) {
            throw new InvalidArgumentException('Minimum density must be'
                . " greater than 0, $minDensity given.");
        }

        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('Max leaf size cannot be'
                . " less than 1, $maxLeafSize given.");
        }

        $this->radius = $radius;
        $this->minDensity = $minDensity;
        $this->kernel = $kernel ?? new Euclidean();
        $this->maxLeafSize = $maxLeafSize;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLUSTERER;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
        ];
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $samples = $dataset->samples();

        $dataset = Labeled::quick($samples, array_keys($samples));

        $tree = new BallTree($this->maxLeafSize, $this->kernel);

        $tree->grow($dataset);
        
        $cluster = self::START_CLUSTER;

        $predictions = [];

        foreach ($samples as $i => $sample) {
            if (isset($predictions[$i])) {
                continue 1;
            }

            [$neighbors, $distances] = $tree->range($sample, $this->radius);

            if (count($neighbors) < $this->minDensity) {
                $predictions[$i] = self::NOISE;

                continue 1;
            }

            $predictions[$i] = $cluster;

            while ($neighbors) {
                $index = array_pop($neighbors);

                if (isset($predictions[$index])) {
                    if ($predictions[$index] === self::NOISE) {
                        $predictions[$index] = $cluster;
                    }

                    continue 1;
                }

                $predictions[$index] = $cluster;

                [$seeds, $distances] = $tree->range($samples[$index], $this->radius);

                if (count($seeds) >= $this->minDensity) {
                    $neighbors = array_unique(array_merge($neighbors, $seeds));
                }
            }

            $cluster++;
        }

        return $predictions;
    }
}
