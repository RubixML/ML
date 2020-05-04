<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Average;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Trees\ExtraTree;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Extra Tree Regressor
 *
 * An *Extremely Randomized* Regression Tree. These trees differ from standard Regression
 * Trees in that they choose candidate splits at random, rather than searching the entire
 * column for the best split. Extra Trees are faster to build and their predictions have
 * higher variance than a regular decision tree.
 *
 * References:
 * [1] P. Geurts. et al. (2005). Extremely Randomized Trees.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ExtraTreeRegressor extends ExtraTree implements Estimator, Learner, RanksFeatures, Persistable
{
    use PredictsSingle;
    
    /**
     * @param int $maxDepth
     * @param int $maxLeafSize
     * @param int|null $maxFeatures
     * @param float $minPurityIncrease
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $maxDepth = PHP_INT_MAX,
        int $maxLeafSize = 3,
        ?int $maxFeatures = null,
        float $minPurityIncrease = 1e-7
    ) {
        parent::__construct($maxDepth, $maxLeafSize, $maxFeatures, $minPurityIncrease);
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * Return the data types that the model is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::categorical(),
            DataType::continuous(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'max_depth' => $this->maxDepth,
            'max_leaf_size' => $this->maxLeafSize,
            'max_features' => $this->maxFeatures,
            'min_purity_increase' => $this->minPurityIncrease,
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
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $this->grow($dataset);
    }

    /**
     * Make a prediction based on the value of a terminal node in the tree.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return (int|float)[]
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            $node = $this->search($sample);

            $predictions[] = $node instanceof Average
                ? $node->outcome()
                : NAN;
        }

        return $predictions;
    }

    /**
     * Terminate the branch with the most likely outcome.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Outcome
     */
    protected function terminate(Labeled $dataset) : Outcome
    {
        [$mean, $variance] = Stats::meanVar($dataset->labels());

        return new Average($mean, $variance, $dataset->numRows());
    }

    /**
     * Compute the impurity of a labeled dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return float
     */
    protected function impurity(Labeled $dataset) : float
    {
        return Stats::variance($dataset->labels());
    }
}
