<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Graph\BinaryNode;
use Rubix\ML\Graph\BinaryTree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

class DecisionTree extends BinaryTree implements Multiclass, Probabilistic, Persistable
{
    /**
     * The maximum depth of a branch before it is forced to terminate.
     *
     * @var int
     */
    protected $maxDepth;

    /**
     * The minimum number of samples that form a consensus to make a prediction.
     *
     * @var int
     */
    protected $minSamples;

    /**
     * The amount of gini impurity to tolerate when choosing a perfect split.
     *
     * @var float
     */
    protected $tolerance;

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
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
     * @param  float  $tolerance
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $minSamples = 5,
                                float $tolerance = 1e-2)
    {
        if ($minSamples < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . ' to make a decision.');
        }

        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have depth less'
                . ' than 1.');
        }

        if ($tolerance < 0 or $tolerance > 1) {
            throw new InvalidArgumentException('Gini tolerance must be between'
                . ' 0 and 1.');
        }

        $this->maxDepth = $maxDepth;
        $this->minSamples = $minSamples;
        $this->tolerance = $tolerance;
        $this->splits = 0;
    }

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
     * Train the decision tree by learning the most optimal splits in the
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

        $this->classes = $dataset->possibleOutcomes();
        $this->columnTypes = $dataset->columnTypes();

        $data = $dataset->samples();

        foreach ($data as $index => &$sample) {
            array_push($sample, $dataset->label($index));
        }

        $this->setRoot($this->findBestSplit($data));
        $this->splits = 1;

        $this->split($this->root);
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
            $predictions[] = $this->search($sample)->get('class');
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $probabilities = [];

        foreach ($dataset as $sample) {
            $probabilities[] = $this->search($sample)->get('probabilities');
        }

        return $probabilities;
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
     * @param  \Rubix\ML\BinaryNode  $root
     * @return \Rubix\ML\BinaryNode
     */
    protected function _search(array $sample, BinaryNode $root) : BinaryNode
    {
        if ($root->terminal) {
            return $root;
        }

        if ($root->type === self::CATEGORICAL) {
            if ($sample[$root->index] === $root->value) {
                return $this->_search($sample, $root->left());
            } else {
                return $this->_search($sample, $root->right());
            }
        } else {
            if ($sample[$root->index] < $root->value) {
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
     * @param  \Rubix\ML\BinaryNode  $root
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
     * as determined by its gini index. The algorithm will terminate early if it
     * finds a homogenous split. i.e. a gini score of 0.
     *
     * @param  array  $data
     * @return \Rubix\ML\BinaryNode
     */
    protected function findBestSplit(array $data) : BinaryNode
    {
        $outcomes = array_unique(array_column($data, count($data[0]) - 1));

        $indices = range(0, count($data[0]) - 2);

        shuffle($indices);

        $best = [
            'gini' => INF, 'index' => null, 'value' => null, 'groups' => [],
        ];

        foreach ($indices as $index) {
            foreach ($data as $row) {
                $groups = $this->partition($data, $index, $row[$index]);

                $gini = $this->calculateGini($groups, $outcomes);

                if ($gini < $best['gini']) {
                    $best['gini'] = $gini;
                    $best['index'] = $index;
                    $best['value'] = $row[$index];
                    $best['groups'] = $groups;
                }

                if ($gini <= $this->tolerance) {
                    break 2;
                }
            }
        }

        return new BinaryNode([
            'index' => $best['index'],
            'value' => $best['value'],
            'type' => $this->columnTypes[$best['index']],
            'gini' => $best['gini'],
            'groups' => $best['groups'],
        ]);
    }

    /**
     * Terminate the branch by selecting the outcome with the highest
     * probability.
     *
     * @param  array  $data
     * @return \Rubix\ML\Graph\BinaryNode
     */
    protected function terminate(array $data) : BinaryNode
    {
        $classes = array_column($data, count(current($data)) - 1);

        $probabilities = array_fill_keys($this->classes, 0.0);

        $n = count($classes);

        foreach (array_count_values($classes) as $class => $count) {
            $probabilities[$class] = $count / $n;
        }

        $outcome = array_search(max($probabilities), $probabilities);

        return new BinaryNode([
            'class' => $outcome,
            'probabilities' =>  $probabilities,
            'terminal' => true,
        ]);
    }

    /**
     * Partition a dataset into left and right subsets.
     *
     * @param  array  $data
     * @param  mixed  $index
     * @param  mixed  $value
     * @return array
     */
    protected function partition(array $data, $index, $value) : array
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
     * Calculate the Gini impurity index for a given split.
     *
     * @param  array  $groups
     * @param  array  $outcomes
     * @return float
     */
    protected function calculateGini(array $groups, array $outcomes) : float
    {
        $total = array_sum(array_map('count', $groups));
        $gini = 0.0;

        foreach ($groups as $group) {
            $count = count($group);

            if ($count === 0) {
                continue 1;
            }

            $counts = array_count_values(array_column($group,
                count(current($group)) - 1));

            $score = 0.0;

            foreach ($outcomes as $outcome) {
                if (isset($counts[$outcome])) {
                    $score += ($counts[$outcome] / $count) ** 2;
                }
            }

            $gini += (1.0 - $score) * ($count / $total);
        }

        return $gini;
    }
}
