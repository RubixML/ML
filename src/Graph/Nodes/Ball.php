<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;

use function Rubix\ML\argmax;

/**
 * Ball
 *
 * A node that contains points that fall within a uniform hypersphere a.k.a. *ball*.
 *
 * @internal
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
     * @var list<string|int|float>
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
     * @var array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    protected $groups;

    /**
     * Factory method to build a hypersphere by splitting the dataset into left and right clusters.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return self
     */
    public static function split(Labeled $dataset, Distance $kernel) : self
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

        $radius = max($distances) ?: 0.0;

        $leftCentroid = $dataset->sample(argmax($distances));

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $leftCentroid);
        }

        $rightCentroid = $dataset->sample(argmax($distances));

        $groups = $dataset->spatialSplit($leftCentroid, $rightCentroid, $kernel);

        return new self($center, $radius, $groups);
    }

    /**
     * @param list<string|int|float> $center
     * @param float $radius
     * @param array{Labeled,Labeled} $groups
     */
    public function __construct(array $center, float $radius, array $groups)
    {
        $this->center = $center;
        $this->radius = $radius;
        $this->groups = $groups;
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
     * Return the left and right splits of the training data.
     *
     * @return array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    public function groups() : array
    {
        return $this->groups;
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
     * Remove the left and right splits of the training data.
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
