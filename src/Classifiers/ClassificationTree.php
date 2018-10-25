<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\CART;
use Rubix\ML\Graph\Nodes\Best;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Other\Functions\Argmax;
use InvalidArgumentException;
use RuntimeException;

/**
 * Classification Tree
 *
 * A Leaf Tree-based classifier that minimizes gini impurity to greedily
 * search for the best splits in a training set.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ClassificationTree extends CART implements Learner, Probabilistic, Persistable
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
        parent::__construct($maxDepth, $maxLeafSize);
        
        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException("Tree must consider at least 1"
                . " feature to determine a split, $maxFeatures given.");
        }

        if ($tolerance < 0.) {
            throw new InvalidArgumentException("Impurity tolerance must be 0"
                . " or greater, $tolerance given.");
        }

        $this->maxFeatures = $maxFeatures;
        $this->tolerance = $tolerance;
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
     * Train the Leaf tree by learning the most optimal splits in the
     * training set.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
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
            $node = $this->search($sample);

            $predictions[] = $node instanceof Best
                ? $node->outcome()
                : null;
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
            $node = $this->search($sample);

            $probabilities[] = $node instanceof Best
                ? $node->probabilities()
                : null;
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
                $value = $sample[$index];

                $groups = $dataset->partition($index, $value);

                $gini = $this->gini($groups);

                if ($gini < $bestGini) {
                    $bestValue = $value;
                    $bestIndex = $index;
                    $bestGroups = $groups;
                    $bestGini = $gini;
                }

                if ($gini < $this->tolerance) {
                    break 2;
                }
            }
        }

        return new Comparison($bestValue, $bestIndex, $bestGroups, $bestGini);
    }

    /**
     * Terminate the branch by selecting the class outcome with the highest
     * probability.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return \Rubix\ML\Graph\Nodes\BinaryNode
     */
    protected function terminate(Labeled $dataset) : BinaryNode
    {
        $probabilities = array_fill_keys($this->classes, 0.);

        $n = $dataset->numRows();

        foreach (array_count_values($dataset->labels()) as $class => $count) {
            $probabilities[$class] = $count / $n;
        }

        $prediction = Argmax::compute($probabilities);
    
        $gini = $this->gini([$dataset]);

        return new Best($prediction, $probabilities, $gini, $n);
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
