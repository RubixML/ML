<?php

namespace Rubix\ML\Graph;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Coordinate;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Graph\Nodes\BoundingBox;
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
     * @var \Rubix\ML\Graph\Nodes\Coordinate|null
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

        if ($kernel and !in_array(DataType::CONTINUOUS, $kernel->compatibility())) {
            throw new InvalidArgumentException('Distance kernel must be'
                . ' compatible with continuous data types.');
        }

        $this->maxLeafSize = $maxLeafSize;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * @return \Rubix\ML\Graph\Nodes\Coordinate|null
     */
    public function root() : ?Coordinate
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
        $this->root = Coordinate::split($dataset);

        $stack = [$this->root];

        while ($stack) {
            $current = array_pop($stack);

            if (!$current instanceof Coordinate) {
                continue 1;
            }

            [$left, $right] = $current->groups();

            if ($left->numRows() > $this->maxLeafSize) {
                $node = Coordinate::split($left);
    
                $current->attachLeft($node);

                $stack[] = $node;
            } else {
                $current->attachLeft(Neighborhood::terminate($left));
            }
    
            if ($right->numRows() > $this->maxLeafSize) {
                $node = Coordinate::split($right);
    
                $current->attachRight($node);

                $stack[] = $node;
            } else {
                $current->attachRight(Neighborhood::terminate($right));
            }

            $current->cleanup();
        }
    }

    /**
     * Search the tree for a neighborhood and return an array of samples and
     * labels.
     *
     * @param array $sample
     * @return \Rubix\ML\Graph\Nodes\Neighborhood|null
     */
    public function search(array $sample) : ?Neighborhood
    {
        $current = $this->root;

        while ($current) {
            if ($current instanceof Coordinate) {
                if ($sample[$current->column()] < $current->value()) {
                    $current = $current->left();
                } else {
                    $current = $current->right();
                }

                continue 1;
            }

            if ($current instanceof Neighborhood) {
                return $current;
            }
        }

        return null;
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

        $neighborhood = $this->search($sample);

        if (!$neighborhood) {
            return [[], []];
        }

        $distances = $labels = [];

        $visited = new SplObjectStorage();

        $stack = [$neighborhood];

        while ($stack) {
            $current = array_pop($stack);

            if ($visited->contains($current)) {
                continue 1;
            }

            $visited->attach($current);

            $parent = $current->parent();

            if ($parent) {
                $stack[] = $parent;
            }

            if ($current instanceof Neighborhood) {
                foreach ($current->samples() as $neighbor) {
                    $distances[] = $this->kernel->compute($sample, $neighbor);
                }

                $labels = array_merge($labels, $current->labels());

                array_multisort($distances, $labels);

                continue 1;
            }

            $target = $distances[$k - 1] ?? INF;

            foreach ($current->children() as $child) {
                if ($visited->contains($child)) {
                    continue 1;
                }

                if ($child instanceof BoundingBox) {
                    foreach ($child->box() as $side) {
                        $distance = $this->kernel->compute($sample, $side);

                        if ($distance < $target) {
                            $stack[] = $child;

                            continue 2;
                        }
                    }
                }

                $visited->attach($child);
            }
        }

        $distances = array_slice($distances, 0, $k);
        $labels = array_slice($labels, 0, $k);

        return [$labels, $distances];
    }
}
