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
use RuntimeException;

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
     * @var int|null
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
     * The memoized random column indices.
     *
     * @var array
     */
    protected $indices = [
        //
    ];

    /**
     * @param  int  $maxDepth
     * @param  int  $maxLeafSize
     * @param  int|null  $maxFeatures
     * @param  float  $tolerance
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $maxLeafSize = 3,
                                ?int $maxFeatures = null, float $tolerance = 1e-3)
    {
        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . ' feature to determine a split.');
        }

        if ($tolerance < 0.) {
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
        $this->indices = $dataset->axes();

        $this->grow($dataset);

        $this->indices = [];
    }

    /**
     * Make a prediction based on the value of a terminal node in the tree.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare() === true) {
            throw new RuntimeException('Estimator has not been trained.');
        }

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
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->bare() === true) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = [];

        foreach ($dataset as $sample) {
            $probabilities[] = $this->search($sample)->meta('probabilities');
        }

        return $probabilities;
    }

    /**
     * Greedy algorithm to choose the best split point for a given dataset.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function findBestSplit(Labeled $dataset) : Comparison
    {
        $bestGini = INF;
        $bestIndex = $bestValue = null;
        $bestGroups = [];

        $maxFeatures = $this->maxFeatures
            ?? (int) round(sqrt($dataset->numColumns()));

        shuffle($this->indices);

        foreach (array_slice($this->indices, 0, $maxFeatures) as $index) {
            foreach ($dataset as $sample) {
                $groups = $dataset->partition($index, $sample[$index]);

                $gini = $this->gini($groups);

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
        $probabilities = array_fill_keys($this->classes, 0.);

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
    protected function gini(array $groups) : float
    {
        $n = array_sum(array_map('count', $groups));

        $gini = 0.;

        foreach ($groups as $group) {
            $k = $group->numRows();

            if ($k < 2) {
                continue 1;
            }

            $density = $k / $n;

            $counts = array_count_values($group->labels());

            foreach ($counts as $count) {
                $gini += $density * (1. - ($count / $n) ** 2);
            }
        }

        return $gini;
    }
}
