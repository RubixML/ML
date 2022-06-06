<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Distance;

use function Rubix\ML\argmax;

/**
 * Clique
 *
 * A leaf node that contains all of the points that fall within radius of the node's center.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Clique implements Hypersphere, BinaryNode
{
    /**
     * The dataset stored in the node.
     *
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected \Rubix\ML\Datasets\Labeled $dataset;

    /**
     * The centroid or multivariate mean of the cluster.
     *
     * @var list<string|int|float>
     */
    protected array $center;

    /**
     * The radius of the cluster centroid.
     *
     * @var float
     */
    protected float $radius;

    /**
     * Terminate a branch with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return self
     */
    public static function terminate(Labeled $dataset, Distance $kernel) : self
    {
        $center = [];

        foreach ($dataset->features() as $column => $values) {
            if ($dataset->featureType($column)->isContinuous()) {
                $center[] = Stats::mean($values);
            } else {
                $center[] = argmax(array_count_values($values));
            }
        }

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }

        $radius = max($distances) ?: 0.0;

        return new self($dataset, $center, $radius);
    }

    /**
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param list<string|int|float> $center
     * @param float $radius
     */
    public function __construct(Labeled $dataset, array $center, float $radius)
    {
        $this->dataset = $dataset;
        $this->center = $center;
        $this->radius = $radius;
    }

    /**
     * Return the dataset stored in the node.
     *
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function dataset() : Labeled
    {
        return $this->dataset;
    }

    /**
     * Return the center vector.
     *
     * @return list<string|int|float>
     */
    public function center() : array
    {
        return $this->center;
    }

    /**
     * Return the radius of the centroid.
     *
     * @return float
     */
    public function radius() : float
    {
        return $this->radius;
    }

    /**
     * Does the hypersphere reduce to a single point?
     *
     * @return bool
     */
    public function isPoint() : bool
    {
        return $this->radius === 0.0;
    }

    /**
     * Return the height of the node in the tree.
     *
     * @return int
     */
    public function height() : int
    {
        return 1;
    }
}
