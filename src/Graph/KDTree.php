<?php

namespace Rubix\ML\Graph;

use Rubix\ML\Graph\Nodes\Box;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Hypercube;
use Rubix\ML\Graph\Nodes\Neighborhood;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use SplObjectStorage;

/**
 * K-d Tree
 *
 * A multi-dimensional binary search tree for fast nearest neighbor queries.
 * Each node maintains its own *bounding box* that is used to prune off leaf
 * nodes during search.
 *
 * [1] J. L. Bentley. (1975). Multidimensional Binary Seach Trees Used for
 * Associative Searching.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDTree implements BinaryTree
{
    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Nodes\Hypercube|null
     */
    protected $root;

    /**
     * The maximum number of samples that each neighborhood node can contain.
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
     * @return \Rubix\ML\Graph\Nodes\Hypercube|null
     */
    public function root() : ?Hypercube
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
        $this->root = Hypercube::split($dataset);

        $stack = [$this->root];

        while ($stack) {
            $current = array_pop($stack);

            if (!$current instanceof Hypercube) {
                continue 1;
            }

            [$left, $right] = $current->groups();

            $current->cleanup();

            if ($left->numRows() > $this->maxLeafSize) {
                $stack[] = $node = Hypercube::split($left);
    
                $current->attachLeft($node);
            } else {
                $current->attachLeft(Neighborhood::terminate($left));
            }
    
            if ($right->numRows() > $this->maxLeafSize) {
                $stack[] = $node = Hypercube::split($right);
    
                $current->attachRight($node);
            } else {
                $current->attachRight(Neighborhood::terminate($right));
            }
        }
    }

    /**
     * Run a k nearest neighbors search of every neighborhood and return
     * the labels and distances in a 2-tuple.
     *
     * @param array $sample
     * @param int $k
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function nearest(array $sample, int $k = 1) : array
    {
        if ($k < 1) {
            throw new InvalidArgumentException('The number of nearest'
                . " neighbors must be greater than 0, $k given.");
        }

        $stack = $distances = $labels = [];

        $visited = new SplObjectStorage();

        $current = $this->root;

        while ($current) {
            $stack[] = $current;

            if ($current instanceof Hypercube) {
                if ($sample[$current->column()] < $current->value()) {
                    $current = $current->left();
                } else {
                    $current = $current->right();
                }

                continue 1;
            }

            if ($current instanceof Neighborhood) {
                break 1;
            }
        }

        while ($stack) {
            $current = array_pop($stack);

            if ($current instanceof Hypercube) {
                $radius = $distances[$k - 1] ?? INF;

                foreach ($current->children() as $child) {
                    if (!$visited->contains($child)) {
                        if ($child instanceof Box) {
                            foreach ($child->sides() as $side) {
                                $distance = $this->kernel->compute($sample, $side);

                                if ($distance < $radius) {
                                    $stack[] = $child;

                                    continue 2;
                                }
                            }
                        }

                        $visited->attach($child);
                    }
                }

                $visited->attach($current);

                continue 1;
            }

            if ($current instanceof Neighborhood) {
                foreach ($current->samples() as $neighbor) {
                    $distances[] = $this->kernel->compute($sample, $neighbor);
                }

                $labels = array_merge($labels, $current->labels());

                array_multisort($distances, $labels);

                $visited->attach($current);
            }
        }

        return [
            array_slice($labels, 0, $k),
            array_slice($distances, 0, $k),
        ];
    }
}
