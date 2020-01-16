<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Best;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Graph\Trees\ExtraTree;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Extra Tree Classifier
 *
 * An Extremely Randomized Classification Tree that recursively chooses node splits
 * with the least entropy among a set of *k* (given by max features) completely
 * random split points. Extra Trees are useful in ensembles such as Random Forest or
 * AdaBoost as the *weak* learner or they can be used on their own. The strength of
 * Extra Trees as compared to more greedy decision trees are their computational
 * efficiency and reduced bias.
 *
 * References:
 * [1] P. Geurts et al. (2005). Extremely Randomized Trees.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ExtraTreeClassifier extends ExtraTree implements Estimator, Learner, Probabilistic, Persistable
{
    use PredictsSingle, ProbaSingle;
    
    /**
     * The memoized class outcomes.
     *
     * @var string[]
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
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->bare();
    }

    /**
     * Train the learner with a dataset.
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

        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $this->classes = $dataset->possibleOutcomes();

        $this->grow($dataset);
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            $node = $this->search($sample);

            $predictions[] = $node instanceof Best
                ? $node->outcome()
                : '?';
        }

        return $predictions;
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $template = array_fill_keys($this->classes, 0.);

        $probabilities = [];

        foreach ($dataset->samples() as $sample) {
            $node = $this->search($sample);

            $probabilities[] = $node instanceof Best
                ? array_replace($template, $node->probabilities()) ?? []
                : [];
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

        $max = max($counts);

        $outcome = (string) array_search($max, $counts);

        $probabilities = [];

        foreach ($counts as $class => $count) {
            $probabilities[$class] = $count / $n;
        }

        $p = $n ? $max / $n : 1.;

        $impurity = -($p * log($p));

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

        $entropy = 0.;

        foreach ($counts as $count) {
            $p = $count / $n;

            $entropy -= $p * log($p);
        }

        return $entropy;
    }
}
