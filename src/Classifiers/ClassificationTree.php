<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\CART;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Other\Functions\Argmax;
use InvalidArgumentException;

/**
 * Classification Tree
 *
 * A Decision Tree-based classifier that minimizes gini impurity to greedily
 * search for the best splits in a training set.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ClassificationTree extends CART implements Estimator, Probabilistic, Persistable
{
    /**
     * The maximum number of features to consider when determining a split.
     *
     * @var int
     */
    protected $maxFeatures;

    /**
     * A small amount of impurity to tolerate when choosing a perfect split.
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
     * The memoized random column index array.
     *
     * @var array
     */
    protected $indices = [
        //
    ];

    /**
     * @param  int  $maxDepth
     * @param  int  $maxLeafSize
     * @param  int  $maxFeatures
     * @param  float  $tolerance
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $maxLeafSize = 5,
                            int $maxFeatures = PHP_INT_MAX, float $tolerance = 1e-3)
    {
        if ($maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . ' feature to determine a split.');
        }

        if ($tolerance < 0) {
            throw new InvalidArgumentException('Impurity tolerance must be 0 or'
                . ' greater.');
        }

        $this->maxFeatures = $maxFeatures;
        $this->tolerance = $tolerance;

        parent::__construct($maxDepth, $maxLeafSize);
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
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

        $this->grow($dataset);

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
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function findBestSplit(Labeled $dataset) : Comparison
    {
        $bestGini = INF;
        $bestIndex = $bestValue = null;
        $bestGroups = [];

        shuffle($this->indices);

        foreach (array_slice($this->indices, 0, $this->maxFeatures) as $index) {
            foreach ($dataset as $sample) {
                $groups = $dataset->partition($index, $sample[$index]);

                $gini = $this->calculateGiniImpurity($groups);

                if ($gini < $bestGini) {
                    $bestGini = $gini;
                    $bestIndex = $index;
                    $bestValue = $sample[$index];
                    $bestGroups = $groups;
                }

                if ($gini < $this->tolerance) {
                    break 2;
                }
            }
        }

        return new Comparison($bestIndex, $bestValue, $bestGroups, $bestGini);
    }

    /**
     * Terminate the branch by selecting the class outcome with the highest
     * probability.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return \Rubix\ML\Graph\Nodes\Decision
     */
    protected function terminate(Labeled $dataset) : Decision
    {
        $probabilities = array_fill_keys($this->classes, 0.0);

        $n = $dataset->numRows();

        foreach (array_count_values($dataset->labels()) as $class => $count) {
            $probabilities[$class] = $count / $n;
        }

        $prediction = Argmax::compute($probabilities);

        return new Decision($prediction, [
            'probabilities' => $probabilities,
        ]);
    }

    /**
     * Calculate the Gini impurity index for a given split.
     *
     * @param  array  $groups
     * @return float
     */
    protected function calculateGiniImpurity(array $groups) : float
    {
        $total = array_sum(array_map('count', $groups));

        $gini = 0.0;

        foreach ($groups as $group) {
            $n = $group->numRows();

            if ($n === 0) {
                continue 1;
            }

            $counts = array_count_values($group->labels());

            foreach ($counts as $count) {
                $gini += (1.0 - ($count / $n) ** 2) * ($n / $total);
            }
        }

        return $gini;
    }
}
