<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Depth;
use Rubix\ML\Graph\Nodes\Isolator;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * I-Tree
 *
 * The base Isolation Tree implementation with completely random node splitting.
 *
 * References:
 * [1] F. T. Liu et al. (2008). Isolation Forest.
 * [2] F. T. Liu et al. (2011). Isolation-based Anomaly Detection.
 *
 * @internal
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
    protected const MAX_LEAF_SIZE = 2;

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
    protected $maxHeight;

    /**
     * @param int $maxHeight
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $maxHeight = PHP_INT_MAX)
    {
        if ($maxHeight < 1) {
            throw new InvalidArgumentException('A tree cannot have a depth'
                . " of less than 1, $maxHeight given.");
        }

        $this->maxHeight = $maxHeight;
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

        /** @var list<array{Isolator,int}> */
        $stack = [[$this->root, 1]];

        while ([$current, $depth] = array_pop($stack)) {
            [$left, $right] = $current->groups();

            $current->cleanup();

            ++$depth;

            if ($left->empty() or $right->empty()) {
                $node = Depth::terminate($left->merge($right), $depth);

                $current->attachLeft($node);
                $current->attachRight($node);

                continue;
            }

            if ($depth >= $this->maxHeight) {
                $current->attachLeft(Depth::terminate($left, $depth));
                $current->attachRight(Depth::terminate($right, $depth));

                continue;
            }

            if ($left->numRows() > self::MAX_LEAF_SIZE) {
                $node = Isolator::split($left);

                $current->attachLeft($node);

                $stack[] = [$node, $depth];
            } else {
                $current->attachLeft(Depth::terminate($left, $depth));
            }

            if ($right->numRows() > self::MAX_LEAF_SIZE) {
                $node = Isolator::split($right);

                $current->attachRight($node);

                $stack[] = [$node, $depth];
            } else {
                $current->attachRight(Depth::terminate($right, $depth));
            }
        }
    }

    /**
     * Search the tree for a leaf node.
     *
     * @param list<string|int|float> $sample
     * @return \Rubix\ML\Graph\Nodes\Depth|null
     */
    public function search(array $sample) : ?Depth
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

                continue;
            }

            if ($current instanceof Depth) {
                return $current;
            }
        }

        return null;
    }
}
