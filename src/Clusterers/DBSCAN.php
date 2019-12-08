<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;

use function count;

/**
 * DBSCAN
 *
 * *Density-Based Spatial Clustering of Applications with Noise* is a clustering algorithm
 * able to find non-linearly separable and arbitrarily-shaped clusters given a radius and
 * density constraint. In addition, DBSCAN also has the ability to mark outliers as *noise*
 * and thus can be used as a *quasi* anomaly detector.
 *
 * > **Note**: Noise samples are assigned to the cluster number *-1*.
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
    public const START_CLUSTER = '0';
    
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
     * The spatial tree used to run range searches.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

    /**
     * @param float $radius
     * @param int $minDensity
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @throws \InvalidArgumentException
     */
    public function __construct(float $radius = 0.5, int $minDensity = 5, ?Spatial $tree = null)
    {
        if ($radius <= 0.) {
            throw new InvalidArgumentException('Neighbor radius must be'
                . " greater than 0, $radius given.");
        }

        if ($minDensity <= 0) {
            throw new InvalidArgumentException('Minimum density must be'
                . " greater than 0, $minDensity given.");
        }

        $this->radius = $radius;
        $this->minDensity = $minDensity;
        $this->tree = $tree ?? new BallTree();
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
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $dataset = Labeled::quick($dataset->samples(), range(0, $dataset->numRows() - 1));

        $this->tree->grow($dataset);

        $predictions = [];
        
        $cluster = self::START_CLUSTER;

        foreach ($dataset->samples() as $i => $sample) {
            if (isset($predictions[$i])) {
                continue 1;
            }

            [$samples, $neighbors, $distances] = $this->tree->range($sample, $this->radius);

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

                $neighbor = $dataset[$index];

                [$samples, $seeds, $distances] = $this->tree->range($neighbor, $this->radius);

                if (count($seeds) >= $this->minDensity) {
                    $neighbors = array_unique(array_merge($neighbors, $seeds));
                }
            }

            ++$cluster;
        }

        $this->tree->destroy();

        return $predictions;
    }
}
