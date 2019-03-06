<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;

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
class Cluster extends BinaryNode implements Ball, Leaf
{
    /**
     * The samples that make up the cluster.
     *
     * @var array
     */
    protected $samples;

    /**
     * The labels that make up the cluster.
     *
     * @var array
     */
    protected $labels;

    /**
     * The center or multivariate mean of the cluster.
     *
     * @var array
     */
    protected $center;

    /**
     * The radius of the cluster.
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
        $samples = $dataset->samples();
        $labels = $dataset->labels();

        $center = Matrix::quick($samples)
            ->transpose()
            ->mean()
            ->asArray();

        $distances = [];

        foreach ($samples as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }

        $radius = max($distances);

        return new self($samples, $labels, $center, $radius);
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
        if (empty($samples)) {
            throw new InvalidArgumentException('Cluster cannot be empty');
        }

        if (count($samples) !== count($labels)) {
            throw new InvalidArgumentException('The number of samples'
                . ' must be equal to the number of labels.');
        }

        if (empty($center)) {
            throw new InvalidArgumentException('Center vector must'
                . ' not be empty.');
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
     * Return the samples in the neighborhood.
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
     * @return array
     */
    public function labels() : array
    {
        return $this->labels;
    }

    /**
     * Return the label cooresponding to the ith sample in the cluster.
     *
     * @param int $index
     * @return mixed
     */
    public function label(int $index)
    {
        return $this->labels[$index];
    }
}
