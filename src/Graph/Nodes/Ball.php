<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;
use InvalidArgumentException;

use function Rubix\ML\argmax;

/**
 * Hypersphere
 *
 * Ball Tree split node that represents the centroid of a set of samples
 * and its radius.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Ball implements BinaryNode, Hypersphere
{
    use HasBinaryChildren;

    /**
     * The center or multivariate mean of the centroid.
     *
     * @var (int|float)[]
     */
    protected $center;

    /**
     * The radius of the centroid.
     *
     * @var float
     */
    protected $radius;

    /**
     * The left and right splits of the training data.
     *
     * @var \Rubix\ML\Datasets\Labeled[]
     */
    protected $groups;

    /**
     * Factory method to build a hypersphere by splitting the dataset into
     * left and right clusters.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return self
     */
    public static function split(Labeled $dataset, Distance $kernel) : self
    {
        $center = array_map([Stats::class, 'mean'], $dataset->columns());
            
        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }

        $radius = max($distances);

        $leftCentroid = $dataset->sample(argmax($distances));

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $leftCentroid);
        }

        $rightCentroid = $dataset->sample(argmax($distances));
        
        $groups = $dataset->spatialPartition($leftCentroid, $rightCentroid, $kernel);

        return new self($center, $radius, $groups);
    }

    /**
     * @param array $center
     * @param float $radius
     * @param array $groups
     * @throws \InvalidArgumentException
     */
    public function __construct(array $center, float $radius, array $groups)
    {
        if (empty($center)) {
            throw new InvalidArgumentException('Center vector must'
                . ' not be empty.');
        }

        if ($radius < 0.) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        if (count($groups) !== 2) {
            throw new InvalidArgumentException('The number of groups'
                . ' must be exactly 2.');
        }

        foreach ($groups as $group) {
            if (!$group instanceof Labeled) {
                throw new InvalidArgumentException('Group must be a'
                    . ' Labeled dataset, ' . gettype($group) . ' given.');
            }
        }

        $this->center = $center;
        $this->radius = $radius;
        $this->groups = $groups;
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
     * Return the left and right splits of the training data.
     *
     * @return array
     */
    public function groups() : array
    {
        return $this->groups;
    }

    /**
     * Remove the left and right splits of the training data.
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
