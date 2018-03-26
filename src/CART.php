<?php

namespace Rubix\Engine;

use Rubix\Engine\Graph\Tree;
use Rubix\Engine\Graph\BinaryNode;
use InvalidArgumentException;

class CART extends Tree implements Estimator
{
    /**
     * The minimum number of samples that a node needs to make a prediction.
     *
     * @var int
     */
    protected $minSamples;

    /**
     * The maximum depth of a branch before it is terminated.
     *
     * @var int
     */
    protected $maxDepth;

    /**
     * The number of features in a sample.
     *
     * @var int
     */
    protected $columns;

    /**
     * The number of times the tree splits.
     *
     * @var int
     */
    protected $splits;

    /**
     * @param  int  $minSamples
     * @param  int  $maxDepth
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $minSamples = 5, int $maxDepth = PHP_INT_MAX)
    {
        $this->minSamples = $minSamples;
        $this->maxDepth = $maxDepth;
        $this->columns = 0;
        $this->splits = 0;
    }

    /**
     * The height of the tree. O(V)
     *
     * @return int
     */
    public function height() : int
    {
        return $this->root->height();
    }

    /**
     * The balance factor of the tree. O(V)
     *
     * @return int
     */
    public function balance() : int
    {
        return $this->root->balance();
    }

    /**
     * The complexity of the CART i.e. the number of splits.
     *
     * @return int
     */
    public function complexity() : int
    {
        return $this->splits;
    }

    /**
     * Train the CART model on a labeled dataset.
     *
     * @param  array  $data
     * @return void
     */
    public function train(array $samples, array $outcomes) : void
    {
        if (count($samples) !== count($outcomes)) {
            throw new InvalidArgumentException('The number of samples and outcomes must be equal.');
        }

        $this->columns = count($samples[0]);
        $this->splits = 1;

        foreach ($samples as $i => &$sample) {
            $sample[] = $outcomes[$i];
        }

        $this->root = $this->findBestSplit($samples);

        $this->split($this->root);
    }

    /**
     * Make a prediction on a given sample.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array
    {
        if (count($sample) !== $this->columns) {
            throw new InvalidArgumentException('Input data must have the same number of columns as the training data.');
        }

        $output = $this->_predict($this->root, $sample);

        return [
            'outcome' => $output->value(),
            'certainty' => $output->certainty,
        ];
    }

    /**
     * Recursive function to traverse the tree and return a terminal node.
     *
     * @param  \Rubix\Engine\BinaryNode  $root
     * @param  array  $sample
     * @return \Rubix\Engine\BinaryNode
     */
    protected function _predict(BinaryNode $root, array $sample) : BinaryNode
    {
        if ($root->is_terminal) {
            return $root;
        }

        if ($root->is_continuous) {
            if ($sample[$root->index] < $root->value()) {
                return $this->_predict($root->left(), $sample);
            } else {
                return $this->_predict($root->right(), $sample);
            }
        } else {
            if ($sample[$root->index] === $root->value()) {
                return $this->_predict($root->left(), $sample);
            } else {
                return $this->_predict($root->right(), $sample);
            }
        }
    }

    /**
     * Recursive function to split the training data adding desision nodes along the
     * way. The terminating conditions are a) split would make node responsible
     * for less values than $minSamples or b) the max depth of a branch has been reached.
     *
     * @param  \Rubix\Engine\BinaryNode  $root
     * @param  int  $depth
     * @return void
     */
    protected function split(BinaryNode $root, int $depth = 0) : void
    {
        list($left, $right) = $root->groups;

        $root->remove('groups');

        if (empty($left) || empty($right)) {
            $node = $this->terminate(array_merge($left, $right));

            $root->attachLeft($node);
            $root->attachRight($node);
            return;
        }

        if ($depth >= $this->maxDepth) {
            $root->attachLeft($this->terminate($left));
            $root->attachRight($this->terminate($right));
            return;
        }

        if (count($left) >= $this->minSamples) {
            $root->attachLeft($this->findBestSplit($left));

            $this->splits++;

            $this->split($root->left(), ++$depth);
    	} else {
            $root->attachLeft($this->terminate($left));
        }

        if (count($right) >= $this->minSamples) {
            $root->attachRight($this->findBestSplit($right));

            $this->splits++;

            $this->split($root->right(), ++$depth);
        } else {
            $root->attachRight($this->terminate($right));
        }
    }

    /**
     * Greedy algorithm to chose the best split point for a given set of data
     * as determined by its gini index. Terminate early if found a homogenous
     * split. i.e. a gini score of 0.
     *
     * @param  array  $data
     * @return \Rubix\Engine\BinaryNode
     */
    protected function findBestSplit(array $data) : BinaryNode
    {
        $best = [
            'gini' => INF, 'index' => null,
            'value' => null, 'groups' => [],
        ];

        $outcomes = array_unique(array_column($data, count($data[0]) - 1));

        foreach (range(0, $this->columns - 1) as $index) {
            foreach ($data as $row) {
                $groups = $this->partition($data, $index, $row[$index]);

                $gini = $this->calculateGini($groups, $outcomes);

                if ($gini < $best['gini']) {
                    $best = [
                        'gini' => $gini, 'index' => $index,
                        'value' => $row[$index], 'groups' => $groups,
                    ];
                }

                if ($gini === 0.0) {
                    break 2;
                }
            }
        }

        return new BinaryNode($best['value'], [
            'index' => $best['index'],
            'gini' => $best['gini'],
            'is_continuous' => is_numeric($best['value']),
            'is_terminal' => false,
            'groups' => $best['groups'],
        ]);
    }

    /**
     * Terminate the branch with the most likely outcome.
     *
     * @param  array  $data
     * @return \Rubix\Engine\BinaryNode
     */
    protected function terminate(array $data) : BinaryNode
    {
        $outcomes = array_count_values(array_column($data, count($data[0]) -1));

        $outcome = array_search(max($outcomes), $outcomes);

        return new BinaryNode($outcome, [
            'certainty' => (float) $outcomes[$outcome] / count($data),
            'is_terminal' => true,
        ]);
    }

    /**
     * Partition a dataset into left and right subsets. O(N)
     *
     * @param  array  $data
     * @param  int  $index
     * @param  mixed  $value
     * @return array
     */
    protected function partition(array $data, int $index, $value) : array
    {
        $left = $right = [];

        foreach ($data as $row) {
            if (is_numeric($row[$index])) {
                if ($row[$index] < $value) {
                    $left[] = $row;
                } else {
                    $right[] = $row;
                }
            } else {
                if ($row[$index] !== $value) {
                    $left[] = $row;
                } else {
                    $right[] = $row;
                }
            }
        }

        return [$left, $right];
    }

    /**
     * Calculate the Gini index for a given split.
     *
     * @param  array  $groups
     * @param  array  $outcomes
     * @return float
     */
    protected function calculateGini(array $groups, array $outcomes) : float
    {
        $n = array_sum(array_map('count', $groups));
        $gini = 0.0;

        foreach ($groups as $group) {
            $count = count($group);

            if ($count === 0) {
                continue;
            }

            $score = 0.0;
            $occurrences = array_count_values(array_column($group, count($group[0]) - 1));

            foreach ($outcomes as $outcome) {
                if (isset($occurrences[$outcome])) {
                    $score += pow($occurrences[$outcome] / $count, 2);
                }
            }

            $gini += (1.0 - $score) * ($count / $n);
        }

        return $gini;
    }
}
