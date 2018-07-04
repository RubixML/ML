<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Persistable;
use Rubix\ML\Graph\BinaryNode;
use Rubix\ML\Graph\BinaryTree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use MathPHP\Statistics\RandomVariable;
use InvalidArgumentException;

class RegressionTree extends BinaryTree implements Regressor, Persistable
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
     * The maximum number of features to consider when determining a split.
     *
     * @var int
     */
    protected $maxFeatures;

    /**
     * The memoized random column index array.
     *
     * @var array
     */
    protected $indices = [
        //
    ];

    /**
     * The type of each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $columnTypes = [
        //
    ];

    /**
     * The number of times the tree has split. i.e. a comparison is made.
     *
     * @var int
     */
    protected $splits;

    /**
     * @param  int  $maxDepth
     * @param  int  $minSamples
     * @param  int  $maxFeatures
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $minSamples = 5,
                                int $maxFeatures = PHP_INT_MAX)
    {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have depth less'
                . ' than 1.');
        }

        if ($minSamples < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . ' to make a decision.');
        }

        if ($maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . ' feature to determine a split.');
        }

        $this->maxDepth = $maxDepth;
        $this->minSamples = $minSamples;
        $this->maxFeatures = $maxFeatures;

        parent::__construct();

        $this->splits = 0;
    }

    /**
     * The complexity of the tree i.e. the number of splits.
     *
     * @return int
     */
    public function complexity() : int
    {
        return $this->splits;
    }

    /**
     * Train the regression tree by learning the optimal splits in the
     * training set.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $this->indices = $dataset->indices();
        $this->columnTypes = $dataset->columnTypes();

        $data = $dataset->samples();

        foreach ($data as $index => &$sample) {
            array_push($sample, $dataset->label($index));
        }

        $this->setRoot($this->findBestSplit($data));
        $this->splits = 1;

        $this->split($this->root);

        $this->indices = [];
    }

    /**
     * Make a prediction based on the value of a terminal node in the tree.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = $this->search($sample)
                ->get('output');
        }

        return $predictions;
    }

    /**
     * Search the tree for a terminal node.
     *
     * @param  array  $sample
     * @return \Rubix\ML\Graph\BinaryNode
     */
    public function search(array $sample) : BinaryNode
    {
        return $this->_search($sample, $this->root);
    }

    /**
     * Recursive function to traverse the tree and return a terminal node.
     *
     * @param  array  $sample
     * @param  \Rubix\ML\Graph\BinaryNode  $root
     * @return \Rubix\ML\Graph\BinaryNode
     */
    protected function _search(array $sample, BinaryNode $root) : BinaryNode
    {
        if ($root->get('terminal', false)) {
            return $root;
        }

        if ($root->get('type') === self::CATEGORICAL) {
            if ($sample[$root->get('index')] === $root->get('value')) {
                return $this->_search($sample, $root->left());
            } else {
                return $this->_search($sample, $root->right());
            }
        } else {
            if ($sample[$root->get('index')] < $root->get('value')) {
                return $this->_search($sample, $root->left());
            } else {
                return $this->_search($sample, $root->right());
            }
        }
    }

    /**
     * Recursive function to split the training data adding decision nodes along the
     * way. The terminating conditions are a) split would make node responsible
     * for less values than $minSamples or b) the max depth of the branch has been reached.
     *
     * @param  \Rubix\ML\Graph\BinaryNode  $root
     * @param  int  $depth
     * @return void
     */
    protected function split(BinaryNode $root, int $depth = 0) : void
    {
        list($left, $right) = $root->get('groups');

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
     * @return \Rubix\ML\Graph\BinaryNode
     */
    protected function findBestSplit(array $data) : BinaryNode
    {
        $best = [
            'sse' => INF, 'index' => null, 'value' => null, 'groups' => [],
        ];

        shuffle($this->indices);

        foreach (array_slice($this->indices, 0, $this->maxFeatures) as $index) {
            foreach ($data as $row) {
                $groups = $this->partition($data, $index, $row[$index]);

                $sse = 0.0;

                foreach ($groups as $group) {
                    if (count($group) === 0) {
                        continue;
                    }

                    $values = array_column($group, count($group[0]) - 1);

                    $sse += RandomVariable::sumOfSquaresDeviations($values);
                }

                if ($sse < $best['sse']) {
                    $best['sse'] = $sse;
                    $best['index'] = $index;
                    $best['value'] = $row[$index];
                    $best['groups'] = $groups;
                }

                if ($sse === 0.0) {
                    break 2;
                }
            }
        }

        return new BinaryNode([
            'index' => $best['index'],
            'value' => $best['value'],
            'type' => $this->columnTypes[$best['index']],
            'sse' => $best['sse'],
            'groups' => $best['groups'],
        ]);
    }

    /**
     * Terminate the branch with the most likely outcome.
     *
     * @param  array  $data
     * @return \Rubix\ML\Graph\BinaryNode
     */
    protected function terminate(array $data) : BinaryNode
    {
        return new BinaryNode([
            'output' => Average::mean(array_column($data, count($data[0]) - 1)),
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
