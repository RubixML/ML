<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\DecisionTree;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\Terminal;
use InvalidArgumentException;

class ClassificationTree extends DecisionTree implements Multiclass, Probabilistic, Persistable
{
    /**
     * The maximum number of features to consider when determining a split.
     *
     * @var int
     */
    protected $maxFeatures;

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The memoized random column index array.
     *
     * @var array|null
     */
    protected $indices;

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
        if ($maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . ' feature to determine a split.');
        }

        parent::__construct($maxDepth, $minSamples);

        $this->maxFeatures = $maxFeatures;
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
        $this->indices = $dataset->indices();

        $data = $dataset->samples();

        foreach ($data as $index => &$sample) {
            array_push($sample, $dataset->label($index));
        }

        $this->grow($data);

        unset($this->indices);
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
            $predictions[] = $this->search($sample)->outcome();
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
            $probabilities[] = $this->search($sample)->meta('probabilities');
        }

        return $probabilities;
    }

    /**
     * Greedy algorithm to chose the best split point for a given set of data.
     * The algorithm will terminate early if it finds a perfect split. i.e. a
     * gini or sse score of 0.
     *
     * @param  array  $data
     * @return \Rubix\ML\Graph\Nodes\Decision
     */
    protected function findBestSplit(array $data) : Decision
    {
        $best = [
            'gini' => INF, 'index' => null, 'value' => null, 'groups' => [],
        ];

        shuffle($this->indices);

        foreach (array_slice($this->indices, 0, $this->maxFeatures) as $index) {
            foreach ($data as $row) {
                $groups = $this->partition($data, $index, $row[$index]);

                $gini = $this->calculateGini($groups);

                if ($gini < $best['gini']) {
                    $best['gini'] = $gini;
                    $best['index'] = $index;
                    $best['value'] = $row[$index];
                    $best['groups'] = $groups;
                }

                if ($gini === 0.0) {
                    break 2;
                }
            }
        }

        return new Decision($best['index'], $best['value'],
            $best['gini'], $best['groups']);
    }

    /**
     * Terminate the branch by selecting the outcome with the highest
     * probability.
     *
     * @param  array  $data
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Terminal
     */
    protected function terminate(array $data, int $depth) : Terminal
    {
        $classes = array_column($data, count(current($data)) - 1);

        $probabilities = array_fill_keys($this->classes, 0.0);

        $n = count($classes);

        foreach (array_count_values($classes) as $class => $count) {
            $probabilities[$class] = $count / $n;
        }

        $prediction = array_search(max($probabilities), $probabilities);

        return new Terminal($prediction, [
            'probabilities' => $probabilities,
        ]);
    }

    /**
     * Calculate the Gini impurity index for a given split.
     *
     * @param  array  $groups
     * @return float
     */
    protected function calculateGini(array $groups) : float
    {
        $total = array_sum(array_map('count', $groups));

        $gini = 0.0;

        foreach ($groups as $group) {
            $n = count($group);

            if ($n === 0) {
                continue 1;
            }

            $values = array_column($group, count(current($group)) - 1);

            foreach (array_count_values($values) as $count) {
                $gini += (1.0 - ($count / $n) ** 2) * ($n / $total);
            }
        }

        return $gini;
    }
}
