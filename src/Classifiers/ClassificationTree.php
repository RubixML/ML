<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\CART;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\argmax;

/**
 * Classification Tree
 *
 * A binary tree-based learner that minimizes gini impurity as a metric
 * to greedily construct a decision tree used for classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ClassificationTree extends CART implements Estimator, Learner, Probabilistic, Persistable
{
    protected const GINI_TOLERANCE = 1e-3;

    protected const CONTINUOUS_DOWNSAMPLE_RATIO = 0.25;
    
    /**
     * The maximum number of features to consider when determining a split.
     *
     * @var int|null
     */
    protected $maxFeatures;

    /**
     * Should we determine max features on the fly?
     *
     * @var bool
     */
    protected $fitMaxFeatures;

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
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $maxDepth = PHP_INT_MAX,
        int $maxLeafSize = 3,
        float $minImpurityIncrease = 0.,
        ?int $maxFeatures = null
    ) {
        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . " feature to determine a split, $maxFeatures given.");
        }

        parent::__construct($maxDepth, $maxLeafSize, $minImpurityIncrease);

        $this->maxFeatures = $maxFeatures;
        $this->fitMaxFeatures = is_null($maxFeatures);
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

        $n = $dataset->numColumns();

        $this->columns = range(0, $n - 1);

        if ($this->fitMaxFeatures) {
            $this->maxFeatures = (int) round(sqrt($n));
        }

        $this->grow($dataset);

        $this->classes = $dataset->possibleOutcomes();
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
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            $node = $this->search($sample);

            $predictions[] = $node instanceof Outcome
                ? $node->class()
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
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $template = array_fill_keys($this->classes, 0.);

        $probabilities = [];

        foreach ($dataset as $sample) {
            $node = $this->search($sample);

            $probabilities[] = $node instanceof Outcome
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
    protected function split(Labeled $dataset) : Comparison
    {
        $bestImpurity = INF;
        $bestColumn = $bestValue = null;
        $bestGroups = [];

        shuffle($this->columns);

        $columns = array_slice($this->columns, 0, $this->maxFeatures);

        foreach ($columns as $column) {
            $values = $dataset->column($column);

            if ($dataset->columnType($column) === DataType::CONTINUOUS) {
                $k = ceil(count($values) * self::CONTINUOUS_DOWNSAMPLE_RATIO);

                $p = range(0, 100, 100 / $k);

                $values = Stats::percentiles($values, $p);
            } else {
                $values = array_unique($values);
            }

            foreach ($values as $value) {
                $groups = $dataset->partition($column, $value);

                $impurity = $this->splitImpurity($groups);

                if ($impurity < $bestImpurity) {
                    $bestColumn = $column;
                    $bestValue = $value;
                    $bestGroups = $groups;
                    $bestImpurity = $impurity;
                }

                if ($impurity <= self::GINI_TOLERANCE) {
                    break 2;
                }
            }
        }

        return new Comparison(
            $bestColumn,
            $bestValue,
            $bestGroups,
            $bestImpurity
        );
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

        $labels = $dataset->labels();

        $counts = array_count_values($labels);

        $outcome = argmax($counts);

        $probabilities = [];

        foreach ($counts as $class => $count) {
            $probabilities[$class] = $count / $n;
        }

        $impurity = 1. - (max($counts) / $n) ** 2;

        return new Outcome($outcome, $probabilities, $impurity, $n);
    }

    /**
     * Calculate the Gini impurity for a given split.
     *
     * @param array $groups
     * @return float
     */
    protected function splitImpurity(array $groups) : float
    {
        $n = array_sum(array_map('count', $groups));

        $impurity = 0.;

        foreach ($groups as $dataset) {
            $m = $dataset->numRows();

            if ($m <= 1) {
                continue 1;
            }

            $counts = array_count_values($dataset->labels());

            $gini = 0.;
    
            foreach ($counts as $count) {
                $gini += 1. - ($count / $n) ** 2;
            }

            $impurity += ($m / $n) * $gini;
        }

        return $impurity;
    }
}
