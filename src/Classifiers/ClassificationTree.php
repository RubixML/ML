<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Graph\CART;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Best;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Classification Tree
 *
 * A binary tree-based classifier that minimizes gini impurity to greedily
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
    protected $columns = [
        //
    ];

    /**
     * @param int $maxDepth
     * @param int $maxLeafSize
     * @param float $minImpurityIncrease
     * @param int|null $maxFeatures
     * @param float $tolerance
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $maxDepth = PHP_INT_MAX,
        int $maxLeafSize = 3,
        float $minImpurityIncrease = 0.,
        ?int $maxFeatures = null,
        float $tolerance = 1e-3
    ) {
        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . " feature to determine a split, $maxFeatures given.");
        }

        if ($tolerance < 0.) {
            throw new InvalidArgumentException('Impurity tolerance must be 0'
                . " or greater, $tolerance given.");
        }

        $this->maxFeatures = $maxFeatures;
        $this->tolerance = $tolerance;

        parent::__construct($maxDepth, $maxLeafSize, $minImpurityIncrease);
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CATEGORICAL,
            DataType::CONTINUOUS,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->bare();
    }

    /**
     * Train the binary tree by learning the most optimal splits in the
     * training set.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $k = $dataset->numColumns();

        $this->classes = $dataset->possibleOutcomes();
        $this->columns = range(0, $dataset->numColumns() - 1);
        $this->maxFeatures = $this->maxFeatures ?? (int) round(sqrt($k));

        $this->grow($dataset);

        $this->columns = [];
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('The learner has not'
                . ' not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

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
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('The learner has not'
                . ' not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $template = array_fill_keys($this->classes, 0.);

        $probabilities = [];

        foreach ($dataset as $sample) {
            $node = $this->search($sample);

            $probabilities[] = $node instanceof Best
                ? array_replace($template, $node->probabilities())
                : null;
        }

        return $probabilities;
    }

    /**
     * Greedy algorithm to choose the best split point for a given dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function findBestSplit(Labeled $dataset) : Comparison
    {
        $bestGini = INF;
        $bestColumn = $bestValue = null;
        $bestGroups = [];

        shuffle($this->columns);

        foreach (array_slice($this->columns, 0, $this->maxFeatures) as $column) {
            $values = array_unique($dataset->column($column));

            foreach ($values as $value) {
                $groups = $dataset->partition($column, $value);

                $gini = $this->gini($groups);

                if ($gini < $bestGini) {
                    $bestColumn = $column;
                    $bestValue = $value;
                    $bestGroups = $groups;
                    $bestGini = $gini;
                }

                if ($gini < $this->tolerance) {
                    break 2;
                }
            }
        }

        return new Comparison($bestColumn, $bestValue, $bestGroups, $bestGini);
    }

    /**
     * Terminate the branch by selecting the class outcome with the highest
     * probability.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\BinaryNode
     */
    protected function terminate(Labeled $dataset) : BinaryNode
    {
        $n = $dataset->numRows();

        $counts = array_count_values($dataset->labels());

        $outcome = Argmax::compute($counts);

        $probabilities = [];

        foreach ($counts as $class => $count) {
            $probabilities[$class] = $count / $n;
        }
    
        $gini = $this->gini([$dataset]);

        return new Best($outcome, $probabilities, $gini, $n);
    }

    /**
     * Calculate the Gini impurity index for a given split.
     *
     * @param array $groups
     * @return float
     */
    protected function gini(array $groups) : float
    {
        $n = array_sum(array_map('count', $groups));

        $impurity = 0.;

        foreach ($groups as $group) {
            $k = $group->numRows();

            if ($k < 2) {
                continue 1;
            }

            $counts = array_count_values($group->labels());

            $gini = 0;

            foreach ($counts as $count) {
                $gini += 1 - ($count / $n) ** 2;
            }

            $impurity += ($k / $n) * $gini;
        }

        return $impurity;
    }
}
