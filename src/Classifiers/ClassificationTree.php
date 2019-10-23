<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\CART;
use Rubix\ML\Graph\Nodes\Best;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\argmax;

/**
 * Classification Tree
 *
 * A binary tree-based learner that greedily constructs a decision map for classification
 * that minimizes the Gini impurity among the training labels during splitting and within
 * leaf nodes. Classification Trees also serve as the base learner of ensemble methods
 * such as Random Forest and AdaBoost.
 *
 * References:
 * [1] W. Y. Loh. (2011). Classification and Regression Trees.
 * [2] K. Alsabti. et al. (1998). CLOUDS: A Decision Tree Classifier for Large
 * Datasets.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ClassificationTree extends CART implements Estimator, Learner, Probabilistic, Persistable
{
    use PredictsSingle, ProbaSingle;
    
    /**
     * The memoized class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

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
            throw new InvalidArgumentException('Learner requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->classes = $dataset->possibleOutcomes();

        $this->grow($dataset);
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
            throw new RuntimeException('Estimator has not been trained.');
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
     * Terminate the branch by selecting the class outcome with the highest
     * probability.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Outcome
     */
    protected function terminate(Labeled $dataset) : Outcome
    {
        $n = $dataset->numRows();

        $counts = array_count_values($dataset->labels());

        $outcome = argmax($counts);

        $probabilities = [];

        foreach ($counts as $class => $count) {
            $probabilities[$class] = $count / $n;
        }

        $impurity = 1. - (max($counts) / $n) ** 2;

        return new Best($outcome, $probabilities, $impurity, $n);
    }

    /**
     * Compute the impurity of a labeled dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return float
     */
    protected function impurity(Labeled $dataset) : float
    {
        $n = $dataset->numRows();

        if ($n <= 1) {
            return 0.;
        }

        $counts = array_count_values($dataset->labels());

        $gini = 0.;

        foreach ($counts as $count) {
            $gini += 1. - ($count / $n) ** 2;
        }

        return $gini;
    }
}
