<?php

namespace Rubix\Engine\Classifiers;

use Rubix\Engine\Supervised;
use Rubix\Engine\Graph\Tree;
use Rubix\Engine\Persistable;
use Rubix\Engine\Graph\BinaryNode;
use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Labeled;
use InvalidArgumentException;

class DecisionTree extends Tree implements Supervised, Probabilistic, Classifier, Persistable
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
     * The complexity of the decision tree i.e. the number of splits.
     *
     * @return int
     */
    public function complexity() : int
    {
        return $this->splits;
    }

    /**
     * The height of the tree. O(V) because node heights are not memoized.
     *
     * @return int
     */
    public function height() : int
    {
        return $this->root->height();
    }

    /**
     * The balance factor of the tree. O(V) because balance requires height of
     * each node.
     *
     * @return int
     */
    public function balance() : int
    {
        return $this->root->balance();
    }

    /**
     * Train the decision tree by learning the most optimal splits in the
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
            $predictions[] = $this->_predict($sample, $this->root)->value();
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function proba(Dataset $samples) : array
    {
        $probabilities = [];

        foreach ($samples as $sample) {
            $probabilities[] = $this->_predict($sample, $this->root)
                ->get('probabilities', []);
        }

        return $probabilities;
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
     * as determined by its gini index. The algorithm will terminate early if it
     * finds a homogenous split. i.e. a gini score of 0.
     *
     * @param  array  $data
     * @return \Rubix\Engine\BinaryNode
     */
    protected function findBestSplit(array $data) : BinaryNode
    {
        $best = [
            'gini' => INF, 'index' => null, 'value' => null, 'groups' => [],
        ];

        $outcomes = array_unique(array_column($data, count(current($data)) - 1));

        for ($index = 0; $index < count(current($data)) - 1; $index++) {
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
            'type' => $this->columnTypes[$best['index']],
            'gini' => $best['gini'],
            'groups' => $best['groups'],
        ]);
    }

    /**
     * Terminate the branch with the selecting the outcome with the highest
     * probability.
     *
     * @param  array  $data
     * @return \Rubix\Engine\Graph\BinaryNode
     */
    protected function terminate(array $data) : BinaryNode
    {
        $classes = array_column($data, count(current($data)) - 1);

        $n = count($classes);

        $probabilities = [];

        foreach (array_count_values($classes) as $class => $count) {
            $probabilities[$class] = $count / $n;
        }

        $outcome = array_search(max($probabilities), $probabilities);

        return new BinaryNode($outcome, [
            'probabilities' =>  $probabilities,
            'terminal' => true,
        ]);
    }

    /**
     * Partition a dataset into left and right subsets. O(N)
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
