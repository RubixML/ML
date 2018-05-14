<?php

namespace Rubix\Engine\Estimators;

use Rubix\Engine\Graph\Tree;
use MathPHP\Statistics\Average;
use Rubix\Engine\Graph\BinaryNode;
use MathPHP\Statistics\Descriptive;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\Estimators\Predictions\Prediction;
use InvalidArgumentException;

class RegressionTree extends Tree implements Estimator, Regressor, Persistable
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
     * @param  int  $minSamples
     * @param  int  $maxDepth
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $minSamples = 5, int $maxDepth = PHP_INT_MAX)
    {
        if ($minSamples < 1) {
            throw new InvalidArgumentException('At least one sample is required to make a decision.');
        }

        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have depth less than 1.');
        }

        $this->minSamples = $minSamples;
        $this->maxDepth = $maxDepth;
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
     * Train the regression tree by learning the most optimal splits in the training set.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        if ($dataset->outcomeType() !== self::CONTINUOUS) {
            throw new InvalidArgumentException('This estimator only works with continuous outcomes.');
        }

        $this->columnTypes = $dataset->columnTypes();

        $this->root = $this->findBestSplit($dataset->all());
        $this->splits = 1;

        $this->split($this->root);
    }

    /**
     * Make a prediction on a given sample.
     *
     * @param  array  $sample
     * @throws \InvalidArgumentException
     * @return \Rubix\Engine\Estimaotors\Predictions\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        if (count($sample) !== $this->columns()) {
            throw new InvalidArgumentException('Input data must have the same number of columns as the training data.');
        }

        $node = $this->_predict($sample, $this->root);

        return new Prediction($node->value());
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
     * as determined by its variance. The algorithm will terminate early if it
     * finds a perfect split. i.e. a variance score of 0.
     *
     * @param  array  $data
     * @return \Rubix\Engine\BinaryNode
     */
    protected function findBestSplit(array $data) : BinaryNode
    {
        $best = ['variance' => INF, 'index' => null, 'value' => null, 'groups' => []];

        $outcomes = array_column($data, count($data[0]) - 1);

        for ($index = 0; $index < $this->columns() - 1; $index++) {
            foreach ($data as $row) {
                $groups = $this->partition($data, $index, $row[$index]);
                $variance = 0.0;

                foreach ($groups as $group) {
                    if (count($group) === 0) {
                        continue;
                    }

                    $values = array_column($group, count($group[0]) - 1);

                    $variance += Descriptive::populationVariance($values);
                }

                if ($variance < $best['variance']) {
                    $best = [
                        'variance' => $variance, 'index' => $index,
                        'value' => $row[$index], 'groups' => $groups,
                    ];
                }

                if ($variance <= 0.0) {
                    break 2;
                }
            }
        }

        return new BinaryNode($best['value'], [
            'index' => $best['index'],
            'groups' => $best['groups'],
        ]);
    }

    /**
     * Terminate the branch with the most likely outcome.
     *
     * @param  array  $data
     * @return \Rubix\Engine\Graph\BinaryNode
     */
    protected function terminate(array $data) : BinaryNode
    {
        $outcomes = array_column($data, count($data[0]) - 1);

        $mean = Average::mean($outcomes);

        $variance = array_reduce($outcomes, function ($carry, $outcome) use ($mean) {
            return $carry += ($outcome - $mean) ** 2;
        }, 0.0) / count($outcomes);

        return new BinaryNode($mean, [
            'variance' =>  $variance,
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
}
