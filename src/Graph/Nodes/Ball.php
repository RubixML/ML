<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Distance;
use Traversable;

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
class Ball implements Hypersphere, BinaryNode, HasBinaryChildren
{
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
     * The left and right splits of the training data.
     *
     * @var list<\Rubix\ML\Datasets\Labeled>
     */
    protected array $groups;

    /**
     * The left child node.
     *
     * @var \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    protected ?\Rubix\ML\Graph\Nodes\BinaryNode $left = null;

    /**
     * The right child node.
     *
     * @var \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    protected ?\Rubix\ML\Graph\Nodes\BinaryNode $right = null;

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
     * Does the hypersphere reduce to a single point?
     *
     * @return bool
     */
    public function isPoint() : bool
    {
        return $this->radius === 0.0;
    }

    /**
     * Return the left and right splits of the training data.
     *
     * @return list<\Rubix\ML\Datasets\Labeled>
     */
    public function groups() : array
    {
        return $this->groups;
    }

    /**
     * Return the left child node.
     *
     * @return \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    public function left() : ?BinaryNode
    {
        return $this->left;
    }

    /**
     * Return the right child node.
     *
     * @return \Rubix\ML\Graph\Nodes\BinaryNode|null
     */
    public function right() : ?BinaryNode
    {
        return $this->right;
    }

    /**
     * Return the children of this node in a generator.
     *
     * @return \Generator<\Rubix\ML\Graph\Nodes\BinaryNode>
     */
    public function children() : Traversable
    {
        if ($this->left) {
            yield $this->left;
        }

        if ($this->right) {
            yield $this->right;
        }
    }

    /**
     * Return the height of the node in the tree.
     *
     * @return int
     */
    public function height() : int
    {
        return 1 + max($this->left ? $this->left->height() : 0, $this->right ? $this->right->height() : 0);
    }

    /**
     * The balance factor of the node. Negative numbers indicate a lean to the left, positive
     * to the right, and 0 is perfectly balanced.
     *
     * @return int
     */
    public function balance() : int
    {
        return ($this->right ? $this->right->height() : 0) - ($this->left ? $this->left->height() : 0);
    }

    /**
     * Set the left child node.
     *
     * @param \Rubix\ML\Graph\Nodes\BinaryNode $node
     */
    public function attachLeft(?BinaryNode $node = null) : void
    {
        $this->left = $node;
    }

    /**
     * Set the right child node.
     *
     * @param \Rubix\ML\Graph\Nodes\BinaryNode $node
     */
    public function attachRight(?BinaryNode $node = null) : void
    {
        $this->right = $node;
    }

    /**
     * Remove the left and right splits of the training data.
     */
    public function cleanup() : void
    {
        $this->groups = [];
    }
}
