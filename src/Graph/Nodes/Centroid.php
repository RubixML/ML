<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;

/**
 * Centroid
 *
 * Ball Tree split node that represents the centroid of a set of samples
 * and its radius.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Centroid extends BinaryNode implements Ball
{
    /**
     * The center or multivariate mean of the centroid.
     *
     * @var array
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
     * @var \Rubix\ML\Datasets\Dataset[]
     */
    protected $groups;

    /**
     * Factory method to build a centroid node from a labeled dataset
     * using the column with the highest variance as the column for the
     * split.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return self
     */
    public static function split(Labeled $dataset, Distance $kernel) : self
    {
        $center = Matrix::quick($dataset->samples())
            ->transpose()
            ->mean()
            ->asArray();
            
        $distances = [];

        foreach ($dataset as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }

        $radius = max($distances);

        $left = $dataset->row(Argmax::compute($distances));

        $distances = [];

        foreach ($dataset as $sample) {
            $distances[] = $kernel->compute($sample, $left);
        }

        $right = $dataset->row(Argmax::compute($distances));

        $leftSamples = $rightSamples = $leftLabels = $rightLabels = [];

        foreach ($dataset as $i => $sample) {
            $lHat = $kernel->compute($sample, $left);
            $rHat = $kernel->compute($sample, $right);

            if ($lHat < $rHat) {
                $leftSamples[] = $sample;
                $leftLabels[] = $dataset->label((int) $i);
            } else {
                $rightSamples[] = $sample;
                $rightLabels[] = $dataset->label((int) $i);
            }
        }

        return new self($center, $radius, [
            Labeled::quick($leftSamples, $leftLabels),
            Labeled::quick($rightSamples, $rightLabels),
        ]);
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

        if ($radius <= 0.) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        if (count($groups) !== 2) {
            throw new InvalidArgumentException('The number of groups'
                . ' must be exactly 2.');
        }

        foreach ($groups as $group) {
            if (!$group instanceof Labeled) {
                throw new InvalidArgumentException('Groups must be'
                    . ' labeled dataset objects, ' . gettype($group)
                    . ' given.');
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
