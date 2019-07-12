<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\CART;
use Rubix\ML\Graph\Nodes\Average;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Regression Tree
 *
 * A Decision Tree learning algorithm (CART) that performs greedy splitting
 * by minimizing the variance (*impurity*) among decision node splits.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RegressionTree extends CART implements Estimator, Learner, Persistable
{
    protected const VARIANCE_TOLERANCE = 1e-5;

    protected const CONTINUOUS_DOWNSAMPLE_RATIO = 0.25;

    protected const MIN_TEST_SPLITS = 2;
    
    /**
     * The maximum number of feature columns to consider when determining
     * a split.
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
     * @param float $minPurityIncrease
     * @param int|null $maxFeatures
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $maxDepth = PHP_INT_MAX,
        int $maxLeafSize = 3,
        float $minPurityIncrease = 0.,
        ?int $maxFeatures = null
    ) {
        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . " feature to determine a split, $maxFeatures given.");
        }

        parent::__construct($maxDepth, $maxLeafSize, $minPurityIncrease);

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
        return self::REGRESSOR;
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
     * Train the regression tree by learning the optimal splits in the
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

        $this->columns = [];
    }

    /**
     * Make a prediction based on the value of a terminal node in the tree.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
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

            $predictions[] = $node instanceof Average
                ? $node->outcome()
                : null;
        }

        return $predictions;
    }

    /**
     * Greedy algorithm to chose the best split for a given dataset as
     * determined by the variance of the split.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function split(Labeled $dataset) : Comparison
    {
        $bestVariance = INF;
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

                $variance = $this->splitImpurity($groups);

                if ($variance < $bestVariance) {
                    $bestColumn = $column;
                    $bestValue = $value;
                    $bestGroups = $groups;
                    $bestVariance = $variance;
                }

                if ($variance <= self::VARIANCE_TOLERANCE) {
                    break 2;
                }
            }
        }

        return new Comparison(
            $bestColumn,
            $bestValue,
            $bestGroups,
            $bestVariance
        );
    }

    /**
     * Terminate the branch with the most likely outcome.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\BinaryNode
     */
    protected function terminate(Labeled $dataset) : BinaryNode
    {
        [$mean, $variance] = Stats::meanVar($dataset->labels());

        return new Average($mean, $variance, $dataset->numRows());
    }

    /**
     * Calculate the weighted variance for the split.
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

            $impurity += ($m / $n) * Stats::variance($dataset->labels());
        }

        return $impurity;
    }
}
