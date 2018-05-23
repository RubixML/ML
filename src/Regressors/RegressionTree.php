<?php

namespace Rubix\Engine\Regressors;

use Rubix\Engine\Supervised;
use Rubix\Engine\Graph\Tree;
use Rubix\Engine\Persistable;
use MathPHP\Statistics\Average;
use Rubix\Engine\Graph\BinaryNode;
use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Labeled;
use MathPHP\Statistics\Descriptive;
use InvalidArgumentException;

class RegressionTree extends Tree implements Supervised, Regressor, Persistable
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
            throw new InvalidArgumentException('At least one sample is required'
                . ' to make a decision.');
        }

        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have depth less'
                . ' than 1.');
        }

        $this->minSamples = $minSamples;
        $this->maxDepth = $maxDepth;
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
     * Train the regression tree by learning the most optimal splits in the
     * training set.
     *
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Labeled $dataset) : void
    {
        $this->columnTypes = $dataset->columnTypes();

        $data = array_map(function ($sample, $label) {
            return array_merge($sample, (array) $label);
        }, ...$dataset->all());

        $this->root = $this->findBestSplit($data);
        $this->splits = 1;

        $this->split($this->root);
    }

    /**
     * Make a prediction based on the value of a terminal node in the tree.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($samples as $sample) {
            $node = $this->_predict($sample, $this->root);

            $predictions[] = $node->value();
        }

        return $predictions;
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

        if ($root->type === self::CATEGORICAL) {
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

        if (empty($left) or empty($right)) {
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
        $best = [
            'variance' => INF, 'index' => null, 'value' => null, 'groups' => [],
        ];

        $outcomes = array_column($data, count($data[0]) - 1);

        for ($index = 0; $index < count(current($data)) - 1; $index++) {
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

                if ($variance === 0.0) {
                    break 2;
                }
            }
        }

        return new BinaryNode($best['value'], [
            'index' => $best['index'],
            'type' => $this->columnTypes[$best['index']],
            'variance' => $best['variance'],
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
