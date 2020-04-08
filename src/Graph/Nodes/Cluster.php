<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;
use InvalidArgumentException;

use function count;
use function Rubix\ML\argmax;

/**
 * Cluster
 *
 * A leaf node that contains all of the points that fall within radius of the node's center.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Cluster implements BinaryNode, Hypersphere, Leaf
{
    use HasBinaryChildren;
    
    /**
     * The dataset stored in the node.
     *
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected $dataset;

    /**
     * The centroid or multivariate mean of the cluster.
     *
     * @var (string|int|float)[]
     */
    protected $center;

    /**
     * The radius of the cluster centroid.
     *
     * @var float
     */
    protected $radius;

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

        foreach ($dataset->columns() as $column => $values) {
            if ($dataset->columnType($column)->isContinuous()) {
                $center[] = Stats::mean($values);
            } else {
                $center[] = argmax(array_count_values($values));
            }
        }

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }

        $radius = max($distances);

        return new self($dataset, $center, $radius);
    }

    /**
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param (string|int|float)[] $center
     * @param float $radius
     * @throws \InvalidArgumentException
     */
    public function __construct(Labeled $dataset, array $center, float $radius)
    {
        if (count($center) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('Center must be'
                . ' same dimensionality as dataset,'
                . " {$dataset->numColumns()} expected but "
                . count($center) . ' given.');
        }

        if ($radius < 0.0) {
            throw new InvalidArgumentException('Radius must be'
                . " 0 or greater, $radius given.");
        }

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
     * @return (string|int|float)[]
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
}
