<?php

namespace Rubix\Engine;

use Rubix\Engine\Graph\Tree;
use MathPHP\Statistics\Average;
use Rubix\Engine\Graph\BinaryNode;
use MathPHP\Statistics\Descriptive;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Persisters\Persistable;
use InvalidArgumentException;

class CART extends Tree implements Estimator, Classifier, Regression, Persistable
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
     * The number of times the tree has split. i.e. a comparison is made.
     *
     * @var int
     */
    protected $splits;

    /**
     * The type of each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $columnTypes = [
        //
    ];

    /**
     * The output type. i.e. categorical or continuous.
     *
     * @var int
     */
    protected $output;

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
        $this->output = 0;
        $this->splits = 0;
    }

    /**
     * @return int
     */
    public function columns() : int
    {
        return count($this->columnTypes);
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
     * Train the CART by learning the most optimal splits in the training set.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        $this->columnTypes = $dataset->columnTypes();
        $this->output = $dataset->outcomeType();

        $this->root = $this->findBestSplit($dataset->all());
        $this->splits = 1;

        $this->split($this->root);
    }

    /**
     * Make a prediction on a given sample.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        if (count($sample) !== $this->columns()) {
            throw new InvalidArgumentException('Input data must have the same number of columns as the training data.');
        }

        $node = $this->_predict($sample, $this->root);

        if ($node->output === self::CATEGORICAL) {
            return new Prediction($node->value(), [
                'probablity' => $node->probability,
            ]);
        } else {
            return new Prediction($node->value(), [
                'variance' => $node->variance,
            ]);
        }
    }

    /**
     * Recursive function to traverse the tree and return a terminal node.
     *
     * @param  array  $sample
     * @param  \Rubix\Engine\BinaryNode  $root
     * @return \Rubix\Engine\BinaryNode
     */
    protected function _predict(array $sample, BinaryNode $root) : BinaryNode
    {
        if ($root->terminal) {
            return $root;
        }

        if ($this->columnTypes[$root->index] === self::CATEGORICAL) {
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

        for ($index = 0; $index < $this->columns() - 1; $index++) {
            foreach ($data as $row) {
                $groups = $this->partition($data, $index, $row[$index]);

                if ($this->output === self::CATEGORICAL) {
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
        $outcomes = array_column($data, count($data[0]) - 1);

        if ($this->output === self::CATEGORICAL) {
            $counts = array_count_values($outcomes);

            $outcome = array_search(max($counts), $counts);

            $probability = $counts[$outcome] / count($outcomes);

            return new BinaryNode($outcome, [
                'probability' =>  $probability,
                'output' => self::CATEGORICAL,
                'terminal' => true,
            ]);
        } else {
            $mean = Average::mean($outcomes);

            $variance = array_reduce($outcomes, function ($carry, $outcome) use ($mean) {
                return $carry += ($outcome - $mean) ** 2;
            }, 0.0) / count($outcomes);

            return new BinaryNode($mean, [
                'variance' =>  $variance,
                'output' => self::CONTINUOUS,
                'terminal' => true,
            ]);
        }
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
            if ($this->columnTypes[$index] === self::CATEGORICAL) {
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

            $variance += Descriptive::populationVariance(array_column($group, count($group[0]) - 1));
        }

        return $variance;
    }
}
