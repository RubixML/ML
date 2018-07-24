<?php

namespace Rubix\ML\Graph;

use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\BinaryNode;
use InvalidArgumentException;

abstract class DecisionTree implements Tree
{
    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Nodes\Comparison|null
     */
    protected $root;

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
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    abstract protected function findBestSplit(array $data) : Comparison;

    /**
     * Terminate the branch by selecting the most likely outcome as the
     * prediction.
     *
     * @param  array  $data
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Decision
     */
    abstract protected function terminate(array $data, int $depth) : Decision;

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
     * @return \Rubix\ML\Graph\Nodes\Comparison|null
     */
    public function root() : ?Comparison
    {
        return $this->root;
    }

    /**
     * The height of the tree. O(V) because node heights are not memoized.
     *
     * @return int
     */
    public function height() : int
    {
        return isset($this->root) ? $this->root->height() : 0;
    }

    /**
     * The balance factor of the tree. O(V) because balance requires height of
     * each node.
     *
     * @return int
     */
    public function balance() : int
    {
        return isset($this->root) ? $this->root->balance() : 0;
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

    /**
     * Return an array indexed by column number that contains the normalized
     * importance score of that column in determining the overall prediction.
     *
     * @return array
     */
    public function featureImportances() : array
    {
        if ($this->bare()) {
            return [];
        }

        $importances = [];

        foreach ($this->traverse($this->root) as $node) {
            if ($node instanceof Comparison) {
                if (isset($importances[$node->index()])) {
                    $importances[$node->index()] += $node->impurityDecrease();
                } else {
                    $importances[$node->index()] = $node->impurityDecrease();
                }
            }
        }

        $total = array_sum($importances);

        foreach ($importances as &$importance) {
            $importance /= $total;
        }

        arsort($importances);

        return $importances;
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
     * @param  \Rubix\ML\Graph\Nodes\Comparison  $current
     * @param  int  $depth
     * @return void
     */
    protected function split(Comparison $current, int $depth) : void
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
     * @return \Rubix\ML\Graph\Nodes\Decision|null
     */
    public function search(array $sample) : ?Decision
    {
        $current = $this->root;

        while (isset($current)) {
            if ($current instanceof Decision) {
                return $current;
            }

            if ($current instanceof Comparison) {
                if (is_string($current->value())) {
                    if ($sample[$current->index()] === $current->value()) {
                        $current = $current->left();
                    } else {
                        $current = $current->right();
                    }
                } else {
                    if ($sample[$current->index()] < $current->value()) {
                        $current = $current->left();
                    } else {
                        $current = $current->right();
                    }
                }
            }
        }

        return null;
    }

    /**
     * Return an array of all the decision nodes in the tree starting at a
     * given node.
     *
     * @param  \Rubix\ML\Graph\Nodes\BinaryNode  $current
     * @return array
     */
    public function traverse(BinaryNode $current) : array
    {
        if ($current instanceof Decision) {
            return [$current];
        }

        $left = $right = [];

        if ($current->left() !== null) {
            $left = $this->traverse($current->left());
        }

        if ($current->right() !== null) {
            $right = $this->traverse($current->right());
        }

        return array_merge([$current], $left, $right);
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
                if ($row[$index] === $value) {
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
