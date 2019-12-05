<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;
use InvalidArgumentException;

use function count;

/**
 * Cluster
 *
 * A Ball Tree leaf node that contains all of the points that fall
 * within radius of the node's center.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Cluster implements BinaryNode, Hypersphere, Leaf
{
    use HasBinaryChildren;
    
    /**
     * The samples that make up the cluster.
     *
     * @var array[]
     */
    protected $samples;

    /**
     * The labels that make up the cluster.
     *
     * @var array
     */
    protected $labels;

    /**
     * The centroid or multivariate mean of the cluster.
     *
     * @var (int|float)[]
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
        $center = array_map([Stats::class, 'mean'], $dataset->columns());

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }

        $radius = max($distances);

        return new self($dataset->samples(), $dataset->labels(), $center, $radius);
    }

    /**
     * @param array $samples
     * @param array $labels
     * @param array $center
     * @param float $radius
     * @throws \InvalidArgumentException
     */
    public function __construct(array $samples, array $labels, array $center, float $radius)
    {
        if (count($samples) !== count($labels)) {
            throw new InvalidArgumentException('The number of samples'
                . ' must be equal to the number of labels.');
        }

        if (empty($center)) {
            throw new InvalidArgumentException('Center cannot be empty.');
        }

        if ($radius < 0.) {
            throw new InvalidArgumentException('Radius must be'
                . " 0 or greater, $radius given.");
        }

        $this->samples = $samples;
        $this->labels = $labels;
        $this->center = $center;
        $this->radius = $radius;
    }

    /**
     * Return the center vector.
     *
     * @return (int|float)[]
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
     * Return the samples in the cluster.
     *
     * @return array[]
     */
    public function samples() : array
    {
        return $this->samples;
    }

    /**
     * Return the labels in the cluster.
     *
     * @return (int|float|string)[]
     */
    public function labels() : array
    {
        return $this->labels;
    }
}
