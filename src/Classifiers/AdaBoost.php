<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function count;

use const Rubix\ML\EPSILON;

/**
 * AdaBoost
 *
 * Short for *Adaptive Boosting*, this ensemble classifier can improve the performance
 * of an otherwise *weak* classifier by focusing more attention on samples that are
 * harder to classify. It builds an additive model where, at each stage, a new learner
 * is instantiated and trained.
 *
 * > **Note**: The default base classifier is a *Decision Stump* i.e a
 * Classification Tree with a max depth of 1.
 *
 * References:
 * [1] Y. Freund et al. (1996). A Decision-theoretic Generalization of On-line
 * Learning and an Application to Boosting.
 * [2] J. Zhu et al. (2006). Multi-class AdaBoost.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AdaBoost implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use PredictsSingle, ProbaSingle, LoggerAware;
    
    /**
     * The base classifier to be boosted.
     *
     * @var \Rubix\ML\Learner
     */
    protected $base;

    /**
     * The learning rate of the ensemble i.e. the *shrinkage* applied to each step.
     *
     * @var float
     */
    protected $rate;

    /**
     * The ratio of samples to train each weak learner on.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The maximum number of estimators to train in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The minimum change in the training loss necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The ensemble of *weak* classifiers.
     *
     * @var \Rubix\ML\Learner[]
     */
    protected $ensemble = [
        //
    ];

    /**
     * The weight of each training sample in the dataset.
     *
     * @var float[]
     */
    protected $weights = [
        //
    ];

    /**
     * The amount of influence a particular classifier has. i.e. the
     * classifier's ability to make accurate predictions.
     *
     * @var float[]
     */
    protected $influences = [
        //
    ];

    /**
     * The zero vector for the possible class outcomes.
     *
     * @var float[]|null
     */
    protected $classes;

    /**
     * The average training loss at each epoch.
     *
     * @var float[]
     */
    protected $steps = [
        //
    ];

    /**
     * @param \Rubix\ML\Learner|null $base
     * @param float $rate
     * @param float $ratio
     * @param int $estimators
     * @param float $minChange
     * @throws \InvalidArgumentException
     */
    public function __construct(
        ?Learner $base = null,
        float $rate = 1.0,
        float $ratio = 0.8,
        int $estimators = 100,
        float $minChange = 1e-4
    ) {
        if ($base and !$base->type()->isClassifier()) {
            throw new InvalidArgumentException('Base estimator must be a'
                . " classifier, {$base->type()} given.");
        }

        if ($rate < 0.0) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . " than 0, $rate given.");
        }

        if ($ratio <= 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0 and 1, $ratio given.");
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . " 1 estimator, $estimators given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . " than 0, $minChange given.");
        }

        $this->base = $base ?? new ClassificationTree(1);
        $this->rate = $rate;
        $this->ratio = $ratio;
        $this->estimators = $estimators;
        $this->minChange = $minChange;
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'base' => $this->base,
            'rate' => $this->rate,
            'ratio' => $this->ratio,
            'estimators' => $this->estimators,
            'min_change' => $this->minChange,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->ensemble and $this->influences;
    }

    /**
     * Return the calculated weight values of the samples in the last training set.
     *
     * @return float[]
     */
    public function weights() : array
    {
        return $this->weights;
    }

    /**
     * Return the list of influence values for each classifier in the ensemble.
     *
     * @return float[]
     */
    public function influences() : array
    {
        return $this->influences;
    }

    /**
     * Return the training loss at each epoch.
     *
     * @return float[]
     */
    public function steps() : array
    {
        return $this->steps;
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

        if ($this->logger) {
            $this->logger->info('Learner init '
                . Params::stringify($this->params()));
        }

        $this->classes = array_fill_keys($dataset->possibleOutcomes(), 0.0);

        $labels = $dataset->labels();
        
        $n = $dataset->numRows();
        $p = (int) round($this->ratio * $n);
        $k = count($this->classes);

        $threshold = 1.0 - (1.0 / $k);
        $prevLoss = INF;

        $this->ensemble = $this->influences = $this->steps = [];

        $this->weights = array_fill(0, $n, 1 / $n);

        for ($epoch = 1; $epoch <= $this->estimators; ++$epoch) {
            $estimator = clone $this->base;

            $subset = $dataset->randomWeightedSubsetWithReplacement($p, $this->weights);

            $estimator->train($subset);

            $predictions = $estimator->predict($dataset);
            
            $loss = 0.0;

            foreach ($predictions as $i => $prediction) {
                if ($prediction !== $labels[$i]) {
                    $loss += $this->weights[$i];
                }
            }

            $total = array_sum($this->weights) ?: EPSILON;

            $loss /= $total;

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch loss=$loss");
            }

            if (is_nan($loss)) {
                break 1;
            }

            if ($loss >= $threshold) {
                if ($this->logger) {
                    $this->logger->info('Estimator dropped due'
                        . ' to high training loss');
                }

                continue 1;
            }

            $influence = $this->rate
                * (log((1.0 - $loss) / ($loss ?: EPSILON))
                + log($k - 1));

            $this->ensemble[] = $estimator;
            $this->influences[] = $influence;

            if ($loss < EPSILON) {
                break 1;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break 1;
            }

            $step = exp($influence);

            foreach ($predictions as $i => $prediction) {
                if ($prediction !== $labels[$i]) {
                    $this->weights[$i] *= $step;
                }
            }

            $total = array_sum($this->weights) ?: EPSILON;

            foreach ($this->weights as &$weight) {
                $weight /= $total;
            }

            $prevLoss = $loss;
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map('Rubix\ML\argmax', $this->score($dataset));
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        $scores = $this->score($dataset);

        $probabilities = [];

        foreach ($scores as $scores) {
            $total = array_sum($scores) ?: EPSILON;

            $dist = [];

            foreach ($scores as $class => $score) {
                $dist[$class] = $score / $total;
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
    }

    /**
     * Return the influence scores for each sample in the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    protected function score(Dataset $dataset) : array
    {
        if (!$this->ensemble or !$this->influences or !$this->classes) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $scores = array_fill(0, $dataset->numRows(), $this->classes);

        foreach ($this->ensemble as $i => $estimator) {
            $influence = $this->influences[$i];

            foreach ($estimator->predict($dataset) as $j => $prediction) {
                $scores[$j][$prediction] += $influence;
            }
        }

        return $scores;
    }
}
