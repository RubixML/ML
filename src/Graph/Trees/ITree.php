<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Cell;
use Rubix\ML\Graph\Nodes\Isolator;
use InvalidArgumentException;

/**
 * I-Tree
 *
 * The base Isolation Tree implementation with completely random node splitting.
 *
 * References:
 * [1] F. T. Liu et al. (2008). Isolation Forest.
 * [2] F. T. Liu et al. (2011). Isolation-based Anomaly Detection.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ITree implements BinaryTree
{
    /**
     * The maximum size of a leaf node in the tree.
     *
     * @var int
     */
    protected const MAX_LEAF_SIZE = 1;
    
    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Nodes\Isolator|null
     */
    protected $root;

    /**
     * The maximum depth of a branch before it is forced to terminate.
     *
     * @var int
     */
    protected $maxDepth;

    /**
     * Calculate the average path length of an unsuccessful search among n nodes.
     *
     * @param int $n
     * @return float
     */
    public static function c(int $n) : float
    {
        switch (true) {
            case $n > 2:
                return 2.0 * (log($n - 1) + M_EULER) - 2.0 * ($n - 1) / $n;

            case $n === 2:
                return 1.0;

            default:
                return 0.0;
        }
    }

    /**
     * @param int $maxDepth
     * @throws \InvalidArgumentException
     */
    public function __construct(int $maxDepth = PHP_INT_MAX)
    {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have a depth'
                . " of less than 1, $maxDepth given.");
        }

        $this->maxDepth = $maxDepth;
    }

    /**
     * Return the height of the tree i.e. the number of levels.
     *
     * @return int
     */
    public function height() : int
    {
        return $this->root ? $this->root->height() : 0;
    }

    /**
     * Return the balance factor of the tree. A balanced tree will have
     * a factor of 0 whereas an imbalanced tree will either be positive
     * or negative indicating the direction and degree of the imbalance.
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
     * Insert a root node and recursively split the dataset until a
     * terminating condition is met.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function grow(Dataset $dataset) : void
    {
        $this->root = Isolator::split($dataset);

        $stack = [[$this->root, 1]];

        while ($stack) {
            [$current, $depth] = array_pop($stack);

            [$left, $right] = $current->groups();

            $current->cleanup();

            ++$depth;

            if ($depth >= $this->maxDepth) {
                $current->attachLeft(Cell::terminate($left, $depth));
                $current->attachRight(Cell::terminate($right, $depth));
    
                continue 1;
            }
    
            if ($left->numRows() > self::MAX_LEAF_SIZE) {
                $node = Isolator::split($left);

                $stack[] = [$node, $depth];
    
                $current->attachLeft($node);
            } else {
                $current->attachLeft(Cell::terminate($left, $depth));
            }
    
            if ($right->numRows() > self::MAX_LEAF_SIZE) {
                $node = Isolator::split($right);

                $stack[] = [$node, $depth];
    
                $current->attachRight($node);
            } else {
                $current->attachRight(Cell::terminate($right, $depth));
            }
        }
    }

    /**
     * Search the tree for a leaf node.
     *
     * @param (string|int|float)[] $sample
     * @return \Rubix\ML\Graph\Nodes\Cell|null
     */
    public function search(array $sample) : ?Cell
    {
        $current = $this->root;

        while ($current) {
            if ($current instanceof Isolator) {
                $value = $current->value();

                if (is_string($value)) {
                    if ($sample[$current->column()] === $value) {
                        $current = $current->left();
                    } else {
                        $current = $current->right();
                    }
                } else {
                    if ($sample[$current->column()] < $value) {
                        $current = $current->left();
                    } else {
                        $current = $current->right();
                    }
                }

                continue 1;
            }

            if ($current instanceof Cell) {
                return $current;
            }
        }

        return null;
    }
}
