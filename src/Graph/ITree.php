<?php

namespace Rubix\ML\Graph;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Cell;
use Rubix\ML\Graph\Nodes\Isolator;
use InvalidArgumentException;

/**
 * I-Tree
 *
 * The base Isloation Tree implementation with completely random splitting.
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
     * The maximum number of samples that a leaf node can contain.
     *
     * @var int
     */
    protected $maxLeafSize;

    /**
     * @param int $maxDepth
     * @param int $maxLeafSize
     * @throws \InvalidArgumentException
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $maxLeafSize = 3)
    {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have a depth'
                . " less than 1, $maxDepth given.");
        }

        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . " to create a leaf, $maxLeafSize given.");
        }

        $this->maxDepth = $maxDepth;
        $this->maxLeafSize = $maxLeafSize;
    }

    /**
     * Return the root node of the tree.
     *
     * @return \Rubix\ML\Graph\Nodes\Isolator|null
     */
    public function root() : ?Isolator
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function grow(Dataset $dataset) : void
    {
        $depth = 1;

        $this->root = Isolator::split($dataset);

        $stack = [[$this->root, $depth]];

        while ($stack) {
            [$current, $depth] = array_pop($stack) ?? [];

            [$left, $right] = $current->groups();

            $depth++;

            if ($depth >= $this->maxDepth) {
                $current->attachLeft(Cell::terminate($left, $depth));
                $current->attachRight(Cell::terminate($right, $depth));
    
                continue 1;
            }
    
            if ($left->numRows() > $this->maxLeafSize) {
                $node = Isolator::split($left);
    
                $current->attachLeft($node);
    
                $stack[] = [$node, $depth];
            } else {
                $current->attachLeft(Cell::terminate($left, $depth));
            }
    
            if ($right->numRows() > $this->maxLeafSize) {
                $node = Isolator::split($right);
    
                $current->attachRight($node);
    
                $stack[] = [$node, $depth];
            } else {
                $current->attachRight(Cell::terminate($right, $depth));
            }

            $current->cleanup();
        }
    }

    /**
     * Search the tree for a terminal node.
     *
     * @param array $sample
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
