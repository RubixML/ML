<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildrenTrait;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\argmax;
use function array_count_values;
use function max;

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
class Ball implements Hypersphere, HasBinaryChildren
{
    use HasBinaryChildrenTrait;

    /**
     * The center or multivariate mean of the centroid.
     *
     * @var list<string|int|float>
     */
    protected array $center;

    /**
     * The radius of the centroid.
     *
     * @var float
     */
    protected float $radius;

    /**
     * The left and right subsets of the training data.
     *
     * @var array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    protected array $subsets;

    /**
     * Factory method to build a hypersphere by splitting the dataset into left and right clusters.
     *
     * @param Labeled $dataset
     * @param Distance $kernel
     * @return self
     */
    public static function split(Labeled $dataset, Distance $kernel) : self
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

        $leftCentroid = $dataset->sample(argmax($distances));

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $leftCentroid);
        }

        $rightCentroid = $dataset->sample(argmax($distances));

        $subsets = $dataset->spatialSplit($leftCentroid, $rightCentroid, $kernel);

        return new self($center, $radius, $subsets);
    }

    /**
     * @param list<string|int|float> $center
     * @param float $radius
     * @param array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled} $subsets
     */
    public function __construct(array $center, float $radius, array $subsets)
    {
        $this->center = $center;
        $this->radius = $radius;
        $this->subsets = $subsets;
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
     * Return the left and right subsets of the training data.
     *
     * @throws RuntimeException
     * @return array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    public function subsets() : array
    {
        if (!isset($this->subsets)) {
            throw new RuntimeException('Subsets property does not exist.');
        }

        return $this->subsets;
    }

    /**
     * Remove any variables carried over from the parent node.
     */
    public function cleanup() : void
    {
        unset($this->subsets);
    }
}
