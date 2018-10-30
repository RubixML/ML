<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Coordinate;
use Rubix\ML\Graph\Nodes\Neighborhood;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;

/**
 * K-d Tree
 *
 * A multidimensional binary search tree or *K-d* Tree is a structure for fast
 * retrieval (log n) by associative search.
 *
 * [1] J. L. Bentley. (1975). Multidimensional Binary Seach Trees Used for
 * Associative Searching.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDTree implements Tree
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
     * The number of dimensions this tree encodes.
     *
     * @var int|null
     */
    protected $dimensionality;

    /**
     * @param  int  $maxLeafSize
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxLeafSize = 10)
    {
        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException("At least one sample is required"
                . " to form a neighborhood, $maxLeafSize given.");
        }

        $this->maxLeafSize = $maxLeafSize;
    }

    /**
     * @return \Rubix\ML\Graph\Nodes\Coordinate|null
     */
    public function root() : ?Coordinate
    {
        return $this->root;
    }

    /**
     * Insert a root node into the tree and recursively split the training data
     * until a terminating condition is met.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function grow(Dataset $dataset) : void
    {
        $this->dimensionality = $dataset->numColumns();

        $this->root = $this->findBestSplit($dataset, 1);

        $this->split($this->root, 1);
    }

    /**
     * Recursive function to split the training data adding coordinate nodes along
     * the way.
     *
     * @param  \Rubix\ML\Graph\Nodes\Coordinate  $current
     * @param  int  $depth
     * @return void
     */
    protected function split(Coordinate $current, int $depth) : void
    {
        list($left, $right) = $current->groups();

        $current->cleanup();

        if ($left->numRows() > $this->maxLeafSize) {
            $node = $this->findBestSplit($left, $depth);

            $current->attachLeft($node);

            $this->split($node, $depth + 1);
        } else {
            $current->attachLeft(new Neighborhood($left));
        }

        if ($right->numRows() > $this->maxLeafSize) {
            $node = $this->findBestSplit($right, $depth);

            $current->attachRight($node);

            $this->split($node, $depth + 1);
        } else {
            $current->attachRight(new Neighborhood($right));
        }
    }

    /**
     * Search the tree for a neighborhood and return an array of samples and
     * labels.
     *
     * @param  array  $sample
     * @return \Rubix\ML\Graph\Nodes\Neighborhood|null
     */
    public function search(array $sample) : ?Neighborhood
    {
        $current = $this->root;

        while (isset($current)) {
            if ($current instanceof Coordinate) {
                if ($sample[$current->index()] < $current->value()) {
                    $current = $current->left();
                } else {
                    $current = $current->right();
                }
            }

            if ($current instanceof Neighborhood) {
                return $current;
            }
        }

        return null;
    }

    /**
     * Find the best split of a given subset of the training data.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Coordinate
     */
    protected function findBestSplit(Dataset $dataset, int $depth) : Coordinate
    {
        $index = $depth % $this->dimensionality;

        $value = Stats::median($dataset->column($index));

        $groups = $dataset->partition($index, $value);

        return new Coordinate($value, $index, $groups);
    }

    /**
     * Is the tree bare?
     *
     * @return bool
     */
    public function bare() : bool
    {
        return is_null($this->root);
    }
}
