<?php

namespace Rubix\ML\Graph\Trees;

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
    protected const CELL_MAX = 1;
    
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
        if ($n <= 1) {
            return 1.;
        }
        
        return 2. * (log($n - 1) + M_EULER) - 2. * ($n - 1) / $n;
    }

    /**
     * @param int $maxDepth
     * @throws \InvalidArgumentException
     */
    public function __construct(int $maxDepth = PHP_INT_MAX)
    {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have a depth'
                . " less than 1, $maxDepth given.");
        }

        $this->maxDepth = $maxDepth;
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
     * Insert a root node and recursively split the dataset a terminating
     * condition is met.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function grow(Dataset $dataset) : void
    {
        $this->root = Isolator::split($dataset);

        $stack = [[$this->root, 1]];

        while ($stack) {
            [$current, $depth] = array_pop($stack) ?? [];

            [$left, $right] = $current->groups();

            $current->cleanup();

            $depth++;

            if ($depth >= $this->maxDepth) {
                $current->attachLeft(Cell::terminate($left, $depth));
                $current->attachRight(Cell::terminate($right, $depth));
    
                continue 1;
            }
    
            if ($left->numRows() > self::CELL_MAX) {
                $node = Isolator::split($left);
    
                $current->attachLeft($node);
    
                $stack[] = [$node, $depth];
            } else {
                $current->attachLeft(Cell::terminate($left, $depth));
            }
    
            if ($right->numRows() > self::CELL_MAX) {
                $node = Isolator::split($right);
    
                $current->attachRight($node);
    
                $stack[] = [$node, $depth];
            } else {
                $current->attachRight(Cell::terminate($right, $depth));
            }
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

    /**
     * Destroy the tree.
     */
    public function destroy() : void
    {
        unset($this->root);
    }
}
