<?php

namespace Rubix\ML\Graph;

use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\Terminal;
use Rubix\ML\Graph\Trees\BinaryTree;
use InvalidArgumentException;

abstract class DecisionTree extends BinaryTree
{
    /**
     * The maximum depth of a branch before it is forced to terminate.
     *
     * @var int
     */
    protected $maxDepth;

    /**
     * The minimum number of samples that each node must contain in order to
     * form a consensus to make a prediction.
     *
     * @var int
     */
    protected $minSamples;

    /**
     * The number of times the tree has split. i.e. a decision is made.
     *
     * @var int
     */
    protected $splits;

    /**
     * @param  int  $maxDepth
     * @param  int  $minSamples
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth, int $minSamples)
    {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have depth less'
                . ' than 1.');
        }

        if ($minSamples < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . ' to make a decision.');
        }

        $this->maxDepth = $maxDepth;
        $this->minSamples = $minSamples;
    }

    /**
     * Greedy algorithm to chose the best split point for a given set of data.
     *
     * @param  array  $data
     * @return \Rubix\ML\Graph\Nodes\Decision
     */
    abstract protected function findBestSplit(array $data) : Decision;

    /**
     * Terminate the branch by selecting the most likely outcome as the
     * prediction.
     *
     * @param  array  $data
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Terminal
     */
    abstract protected function terminate(array $data, int $depth) : Terminal;

    /**
     * The complexity of the decision tree i.e. the number of splits.
     *
     * @return int
     */
    public function complexity() : int
    {
        return $this->splits;
    }

    /**
     * Insert a root node into the tree and recursively split the training data
     * until a terminating condition is met.
     *
     * @param  array  $data
     * @return void
     */
    public function grow(array $data) : void
    {
        $this->root = $this->findBestSplit($data);

        $this->splits = 1;

        $this->split($this->root, 1);
    }

    /**
     * Recursive function to split the training data adding decision nodes along
     * the way. The terminating conditions are a) split would make node
     * responsible for less values than $minSamples or b) the max depth of the
     * branch has been reached.
     *
     * @param  \Rubix\ML\Graph\Nodes\Decision  $current
     * @param  int  $depth
     * @return void
     */
    protected function split(Decision $current, int $depth) : void
    {
        list($left, $right) = $current->groups();

        $current->cleanup();

        if (empty($left) or empty($right)) {
            $node = $this->terminate(array_merge($left, $right), $depth);

            $current->attachLeft($node);
            $current->attachRight($node);
            return;
        }

        if ($depth >= $this->maxDepth) {
            $current->attachLeft($this->terminate($left, $depth));
            $current->attachRight($this->terminate($right, $depth));
            return;
        }

        if (count($left) > $this->minSamples) {
            $node = $this->findBestSplit($left);

            $current->attachLeft($node);

            $this->splits++;

            $this->split($node, $depth + 1);
        } else {
            $current->attachLeft($this->terminate($left, $depth));
        }

        if (count($right) > $this->minSamples) {
            $node = $this->findBestSplit($right);

            $current->attachRight($node);

            $this->splits++;

            $this->split($node, $depth + 1);
        } else {
            $current->attachRight($this->terminate($right, $depth));
        }
    }

    /**
     * Search the tree for a terminal node.
     *
     * @param  array  $sample
     * @return \Rubix\ML\Graph\Nodes\Terminal|null
     */
    public function search(array $sample) : ?Terminal
    {
        $current = $this->root;

        while (isset($current)) {
            if ($current instanceof Terminal) {
                break 1;
            }

            $value = $current->value();

            if (is_string($value)) {
                if ($sample[$current->index()] === $value) {
                    $current = $current->left();
                } else {
                    $current = $current->right();
                }
            } else {
                if ($sample[$current->index()] < $value) {
                    $current = $current->left();
                } else {
                    $current = $current->right();
                }
            }
        }

        return $current;
    }

    /**
     * Partition a dataset into left and right subsets.
     *
     * @param  array  $data
     * @param  int  $index
     * @param  mixed  $value
     * @return array
     */
    protected function partition(array $data, int $index, $value) : array
    {
        $left = $right = [];

        if (is_string($value)) {
            foreach ($data as $row) {
                if ($row[$index] !== $value) {
                    $left[] = $row;
                } else {
                    $right[] = $row;
                }
            }
        } else {
            foreach ($data as $row) {
                if ($row[$index] < $value) {
                    $left[] = $row;
                } else {
                    $right[] = $row;
                }
            }
        }

        return [$left, $right];
    }
}
