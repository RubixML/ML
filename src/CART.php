<?php

namespace Rubix\Engine;

use Rubix\Engine\Graph\Tree;
use Rubix\Engine\Graph\BinaryNode;
use MathPHP\Statistics\Descriptive;
use InvalidArgumentException;

class CART extends Tree implements Classifier, Regression
{
    /**
     * The minimum number of samples that form a consensus to make a prediction.
     *
     * @var int
     */
    protected $minSamples;

    /**
     * The maximum depth of a branch before it is forced to terminate.
     *
     * @var int
     */
    protected $maxDepth;

    /**
     * Is this a classifier model?
     *
     * @var bool|null
     */
    protected $classifier;

    /**
     * The number of feature columns per sample.
     *
     * @var int
     */
    protected $columns;

    /**
     * The number of times the tree has split. i.e. a comparison is made.
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
        $this->classifier = null;
        $this->columns = 0;
        $this->splits = 0;
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
     * The height of the tree. O(V) because heights are not memoized.
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
     * Is this model a classifier?
     *
     * @return bool|null
     */
    public function classifier() : ?bool
    {
        return $this->classifier;
    }

    /**
     * Does this model output continuous data?
     *
     * @return bool|null
     */
    public function regression() : ?bool
    {
        return !$this->classifier;
    }

    /**
     * Train the CART by learning the most optimal splits in the training set.
     *
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return void
     */
    public function train(SupervisedDataset $data) : void
    {
        list($samples, $outcomes) = $data->toArray();

        $this->columns = $data->columns();
        $this->classifier = is_string($outcomes[0]);

        foreach ($samples as $i => &$sample) {
            $sample[] = $outcomes[$i];
        }

        $this->root = $this->findBestSplit($samples);
        $this->splits = 1;

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

        $output = $this->_predict($sample, $this->root);

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
    protected function _predict(array $sample, BinaryNode $root) : BinaryNode
    {
        if ($root->terminal) {
            return $root;
        }

        if ($root->categorical) {
            if ($sample[$root->index] === $root->value()) {
                return $this->_predict($sample, $root->left());
            } else {
                return $this->_predict($sample, $root->right());
            }
        } else {
            if ($sample[$root->index] < $root->value()) {
                return $this->_predict($sample, $root->left());
            } else {
                return $this->_predict($sample, $root->right());
            }
        }
    }

    /**
     * Recursive function to split the training data adding decision nodes along the
     * way. The terminating conditions are a) split would make node responsible
     * for less values than $minSamples or b) the max depth of the branch has been reached.
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
     * as determined by its gini index, or variance for continuous data. The
     * algorithm will terminate early if it finds a homogenous split. i.e. a gini
     * or variance score of 0.
     *
     * @param  array  $data
     * @return \Rubix\Engine\BinaryNode
     */
    protected function findBestSplit(array $data) : BinaryNode
    {
        $best = [
            'cost' => INF, 'index' => null,
            'value' => null, 'groups' => [],
        ];

        $outcomes = array_column($data, count($data[0]) - 1);

        foreach (range(0, $this->columns - 1) as $index) {
            foreach ($data as $row) {
                $groups = $this->partition($data, $index, $row[$index]);

                if ($this->classifier) {
                    $cost = $this->calculateGini($groups, $outcomes);
                } else {
                    $cost = $this->calculateVariance($groups, $outcomes);
                }

                if ($cost < $best['cost']) {
                    $best = [
                        'cost' => $cost, 'index' => $index,
                        'value' => $row[$index], 'groups' => $groups,
                    ];
                }

                if ($cost === 0.0) {
                    break 2;
                }
            }
        }

        return new BinaryNode($best['value'], [
            'index' => $best['index'],
            'cost' => $best['cost'],
            'categorical' => is_string($best['value']),
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
        $outcomes = array_count_values(array_column($data, count($data[0]) - 1));

        $outcome = array_search(max($outcomes), $outcomes);

        return new BinaryNode($outcome, [
            'certainty' => (float) $outcomes[$outcome] / count($data),
            'terminal' => true,
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
            if (is_string($row[$index])) {
                if ($row[$index] !== $value) {
                    $left[] = $row;
                } else {
                    $right[] = $row;
                }
            } else {
                if ($row[$index] < $value) {
                    $left[] = $row;
                } else {
                    $right[] = $row;
                }
            }
        }

        return [$left, $right];
    }

    /**
     * Calculate the Gini index for a given split. Used for categorical data.
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
                continue 1;
            }

            $score = 0.0;
            $occurrences = array_count_values(array_column($group, count($group[0]) - 1));

            foreach (array_unique($outcomes) as $outcome) {
                if (isset($occurrences[$outcome])) {
                    $score += ($occurrences[$outcome] / $count) ** 2;
                }
            }

            $gini += (1.0 - $score) * ($count / $n);
        }

        return $gini;
    }

    /**
     * Calculate the variance of a given split. Used for continuous data.
     *
     * @param  array  $groups
     * @param  array  $outcomes
     * @return float
     */
    protected function calculateVariance(array $groups, array $outcomes) : float
    {
        $variance = 0.0;

        foreach ($groups as $group) {
            if (count($group) === 0) {
                continue;
            }

            $occurrences = array_column($group, count($group[0]) - 1);

            $variance += Descriptive::populationVariance($occurrences);
        }

        return $variance;
    }
}
