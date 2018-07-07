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

class ExtraTree extends DecisionTree implements Multiclass, Probabilistic, Persistable
{
    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param  int  $maxDepth
     * @param  int  $minSamples
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $minSamples = 5)
    {
        parent::__construct($maxDepth, $minSamples);
    }

    /**
     * Train the decision tree by randomly splitting the training set.
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

        $data = $dataset->samples();

        foreach ($data as $index => &$sample) {
            array_push($sample, $dataset->label($index));
        }

        $this->grow($data);
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
     * Randomized algorithm to split the training set into left and right
     * subsets.
     *
     * @param  array  $data
     * @return \Rubix\ML\Graph\Nodes\Decision
     */
    protected function findBestSplit(array $data) : Decision
    {
        $index = random_int(0, count($data[0]) - 2);

        $value = $data[random_int(0, count($data) - 1)][$index];

        $score = count($data);

        $groups = $this->partition($data, $index, $value);

        return new Decision($index, $value, $score, $groups);
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
}
