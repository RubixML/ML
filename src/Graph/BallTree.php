<?php

namespace Rubix\ML\Graph;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Ball;
use Rubix\ML\Graph\Nodes\Cluster;
use Rubix\ML\Graph\Nodes\Centroid;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;

/**
 * Ball Tree
 *
 * A binary spatial tree with *ball* nodes whose boundary is defined by a
 * centroid and radius.
 *
 * References:
 * [1] S. M. Omohundro. (1989). Five Balltree Construction Algorithms.
 * [2] M. Dolatshah et al. (2015). Ball*-tree: Efficient spatial indexing for
 * constrained nearest-neighbor search in metric spaces.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BallTree implements BinaryTree
{
    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Nodes\Centroid|null
     */
    protected $root;

    /**
     * The maximum number of samples that each ball node can contain.
     *
     * @var int
     */
    protected $maxLeafSize;

    /**
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * @param int $maxLeafSize
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \InvalidArgumentException
     */
    public function __construct(int $maxLeafSize = 20, ?Distance $kernel = null)
    {
        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . " to form a neighborhood, $maxLeafSize given.");
        }

        $this->maxLeafSize = $maxLeafSize;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * @return \Rubix\ML\Graph\Nodes\Centroid|null
     */
    public function root() : ?Centroid
    {
        return $this->root;
    }

    /**
     * Return the height of the tree.
     *
     * @return int
     */
    public function height() : int
    {
        return $this->root ? $this->root->height() : 0;
    }

    /**
     * Return the balance of the tree.
     *
     * @return int
     */
    public function balance() : int
    {
        return $this->root ? $this->root->balance() : 0;
    }

    /**
     * Is the tree bare?
     *
     * @return bool
     */
    public function bare() : bool
    {
        return !$this->root;
    }

    /**
     * Insert a root node into the tree and recursively split the training data
     * until a terminating condition is met.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function grow(Labeled $dataset) : void
    {
        $this->root = Centroid::split($dataset, $this->kernel);

        $stack = [$this->root];

        while ($stack) {
            $current = array_pop($stack);

            if (!$current instanceof Centroid) {
                continue 1;
            }

            [$left, $right] = $current->groups();

            if ($left->numRows() > $this->maxLeafSize) {
                $node = Centroid::split($left, $this->kernel);
    
                $current->attachLeft($node);

                $stack[] = $node;
            } else {
                $node = Cluster::terminate($left, $this->kernel);

                $current->attachLeft($node);
            }
    
            if ($right->numRows() > $this->maxLeafSize) {
                $node = Centroid::split($right, $this->kernel);
    
                $current->attachRight($node);

                $stack[] = $node;
            } else {
                $node = Cluster::terminate($right, $this->kernel);

                $current->attachRight($node);
            }

            $current->cleanup();
        }
    }

    /**
     * Run a range search over every cluster within radius and return
     * the labels and distances in a 2-tuple.
     *
     * @param array $sample
     * @param float $radius
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function range(array $sample, float $radius = 1.0) : array
    {
        if ($radius <= 0.) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        $distances = $labels = [];

        $stack = [$this->root];

        while ($stack) {
            $current = array_pop($stack);

            if (!$current instanceof BinaryNode) {
                continue 1;
            }

            if ($current instanceof Cluster) {
                foreach ($current->samples() as $i => $neighbor) {
                    $distance = $this->kernel->compute($sample, $neighbor);

                    if ($distance <= $radius) {
                        $distances[] = $distance;
                        $labels[] = $current->label($i);
                    }
                }

                continue 1;
            }

            foreach ($current->children() as $child) {
                if ($child instanceof Ball) {
                    $distance = $this->kernel->compute($sample, $child->center());

                    if ($distance <= ($child->radius() + $radius)) {
                        $stack[] = $child;
                    }
                }
            }
        }

        return [$labels, $distances];
    }
}
