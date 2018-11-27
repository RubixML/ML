<?php

namespace Rubix\ML\Graph;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Cell;
use Rubix\ML\Graph\Nodes\Isolator;
use InvalidArgumentException;

/**
 * I Tree
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
class ITree implements Tree
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
     * @param  int  $maxDepth
     * @param  int  $maxLeafSize
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $maxLeafSize = 3)
    {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException("A tree cannot have a depth"
                . " less than 1, $maxDepth given.");
        }

        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException("At least one sample is required"
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
     * Insert a root node into the tree and recursively split the training data
     * until a terminating condition is met.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function grow(Dataset $dataset) : void
    {
        $depth = 1;

        $this->root = $this->findBestSplit($dataset, $depth);

        $stack = [[$this->root, $depth]];

        while($stack) {
            list($current, $depth) = array_pop($stack) ?? [];

            list($left, $right) = $current->groups();

            $depth++;

            if ($depth >= $this->maxDepth) {
                $current->attachLeft($this->terminate($left, $depth));
                $current->attachRight($this->terminate($right, $depth));
    
                continue 1;
            }
    
            if ($left->numRows() > $this->maxLeafSize) {
                $node = $this->findBestSplit($left, $depth);
    
                $current->attachLeft($node);
    
                $stack[] = [$node, $depth];
            } else {
                $current->attachLeft($this->terminate($left, $depth));
            }
    
            if ($right->numRows() > $this->maxLeafSize) {
                $node = $this->findBestSplit($right, $depth);
    
                $current->attachRight($node);
    
                $stack[] = [$node, $depth];
            } else {
                $current->attachRight($this->terminate($right, $depth));
            }

            $current->cleanup();
        }
    }

    /**
     * Search the tree for a terminal node.
     *
     * @param  array  $sample
     * @return \Rubix\ML\Graph\Nodes\Cell|null
     */
    public function search(array $sample) : ?Cell
    {
        $current = $this->root;

        while (isset($current)) {
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
     * Randomized algorithm to find a split point in the data.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Isolator
     */
    protected function findBestSplit(Dataset $dataset, int $depth) : Isolator
    {
        $column = rand(0, $dataset->numColumns() - 1);

        $sample = $dataset[rand(0, count($dataset) - 1)];

        $value = $sample[$column];

        $groups = $dataset->partition($column, $value);

        return new Isolator($column, $value, $groups);
    }

    /**
     * Terminate the branch.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Cell
     */
    protected function terminate(Dataset $dataset, int $depth) : Cell
    {
        $depth += $this->c($dataset->numRows()) - 1.;

        return new Cell($depth);
    }

    /**
     * Calculate the average path length of an unsuccessful search for n nodes.
     *
     * @param  int  $n
     * @return float
     */
    protected function c(int $n) : float
    {
        if ($n <= 1) {
            return 1.;
        }
        
        return 2. * (log($n - 1) + M_EULER) - 2. * ($n - 1) / $n;
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