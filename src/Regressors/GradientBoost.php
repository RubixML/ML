<?php

namespace Rubix\ML\Regressors;

use Tensor\Vector;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function is_nan;
use function array_slice;
use function get_class;
use function in_array;

/**
 * Gradient Boost
 *
 * Gradient Boost is a stage-wise additive ensemble that uses a Gradient Descent boosting
 * scheme for training  boosters (Decision Trees) to correct the error residuals of a
 * series of *weak* base learners. Stochastic gradient boosting is achieved by varying
 * the ratio of samples to subsample uniformly at random from the training set.
 *
 * > **Note**: The default base classifier is a Dummy Classifier using the Mean strategy
 * and the default booster is a Regression Tree with a max height of 3.
 *
 * References:
 * [1] J. H. Friedman. (2001). Greedy Function Approximation: A Gradient
 * Boosting Machine.
 * [2] J. H. Friedman. (1999). Stochastic Gradient Boosting.
 * [3] Y. Wei. et al. (2017). Early stopping for kernel boosting algorithms:
 * A general analysis with localized complexities.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GradientBoost implements Estimator, Learner, RanksFeatures, Verbose, Persistable
{
    use AutotrackRevisions, PredictsSingle, LoggerAware;

    /**
     * The class names of the compatible learners to used as boosters.
     *
     * @var class-string[]
     */
    public const COMPATIBLE_BOOSTERS = [
        RegressionTree::class,
        ExtraTreeRegressor::class,
    ];

    /**
     * The minimum size of each training subset.
     *
     * @var int
     */
    protected const MIN_SUBSAMPLE = 1;

    /**
     * The regressor that will fix up the error residuals of the *weak* base learner.
     *
     * @var \Rubix\ML\Learner
     */
    protected $booster;

    /**
     * The learning rate of the ensemble i.e. the *shrinkage* applied to each step.
     *
     * @var float
     */
    protected $rate;

    /**
     * The ratio of samples to subsample from the training set for each booster.
     *
     * @var float
     */
    protected $ratio;

    /**
     *  The max number of estimators to train in the ensemble.
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
     * The number of epochs without improvement in the validation score to wait
     * before considering an early stop.
     *
     * @var int
     */
    protected $window;

    /**
     * The proportion of training samples to use for validation and progress monitoring.
     *
     * @var float
     */
    protected $holdOut;

    /**
     * The metric used to score the generalization performance of the model
     * during training.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Metric
     */
    protected $metric;

    /**
     * The *weak* base regressor to be boosted.
     *
     * @var \Rubix\ML\Learner
     */
    protected $base;

    /**
     * An ensemble of weak regressors.
     *
     * @var mixed[]
     */
    protected $ensemble = [
        //
    ];

    /**
     * The dimensionality of the training set.
     *
     * @var int|null
     */
    protected $featureCount;

    /**
     * The validation scores at each epoch.
     *
     * @var float[]|null
     */
    protected $scores;

    /**
     * The average training loss at each epoch.
     *
     * @var float[]|null
     */
    protected $steps;

    /**
     * @param \Rubix\ML\Learner|null $booster
     * @param float $rate
     * @param float $ratio
     * @param int $estimators
     * @param float $minChange
     * @param int $window
     * @param float $holdOut
     * @param \Rubix\ML\CrossValidation\Metrics\Metric|null $metric
     * @param \Rubix\ML\Learner|null $base
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        ?Learner $booster = null,
        float $rate = 0.1,
        float $ratio = 0.5,
        int $estimators = 1000,
        float $minChange = 1e-4,
        int $window = 10,
        float $holdOut = 0.1,
        ?Metric $metric = null,
        ?Learner $base = null
    ) {
        if ($booster and !in_array(get_class($booster), self::COMPATIBLE_BOOSTERS)) {
            throw new InvalidArgumentException('Booster is not compatible'
                . ' with the ensemble.');
        }

        if ($rate <= 0.0 or $rate > 1.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($ratio <= 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Number of estimators'
                . " must be greater than 0, $estimators given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be'
                . " greater than 0, $window given.");
        }

        if ($holdOut < 0.0 or $holdOut > 0.5) {
            throw new InvalidArgumentException('Hold out ratio must be'
                . " between 0 and 0.5, $holdOut given.");
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::with($this, $metric)->check();
        }

        if ($base and $base->type() != EstimatorType::regressor()) {
            throw new InvalidArgumentException('Base Estimator must be a'
                . " regressor, {$base->type()} given.");
        }

        $this->booster = $booster ?? new RegressionTree(3);
        $this->rate = $rate;
        $this->ratio = $ratio;
        $this->estimators = $estimators;
        $this->minChange = $minChange;
        $this->window = $window;
        $this->holdOut = $holdOut;
        $this->metric = $metric ?? new RMSE();
        $this->base = $base ?? new DummyRegressor(new Mean());
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        $compatibility = array_intersect(
            $this->booster->compatibility(),
            $this->base->compatibility()
        );

        return array_values($compatibility);
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'booster' => $this->booster,
            'rate' => $this->rate,
            'ratio' => $this->ratio,
            'estimators' => $this->estimators,
            'min_change' => $this->minChange,
            'window' => $this->window,
            'hold_out' => $this->holdOut,
            'metric' => $this->metric,
            'base' => $this->base,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->base->trained() and $this->ensemble;
    }

    /**
     * Return the validation scores at each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->scores;
    }

    /**
     * Return the loss at each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function steps() : ?array
    {
        return $this->steps;
    }

    /**
     * Train the estimator with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        if ($this->logger) {
            $this->logger->info("$this initialized");
        }

        $this->featureCount = $dataset->numColumns();

        [$testing, $training] = $dataset->randomize()->split($this->holdOut);

        [$min, $max] = $this->metric->range();

        if ($this->logger) {
            $this->logger->info("Training {$this->base}");
        }

        $this->base->train($training);

        $this->ensemble = $this->scores = $this->steps = [];

        /** @var list<int|float> $predictions */
        $predictions = $this->base->predict($training);

        $out = $prevOut = Vector::quick($predictions);
        $target = Vector::quick($training->labels());

        if (!$testing->empty()) {
            /** @var list<int|float> $predictions */
            $predictions = $this->base->predict($testing);

            $prevPred = Vector::quick($predictions);
        }

        $p = max(self::MIN_SUBSAMPLE, (int) round($this->ratio * $training->numRows()));

        $bestScore = $min;
        $bestEpoch = $delta = 0;
        $score = null;
        $prevLoss = INF;

        for ($epoch = 1; $epoch <= $this->estimators; ++$epoch) {
            $gradient = $target->subtract($out);

            $training = Labeled::quick($training->samples(), $gradient->asArray());

            $booster = clone $this->booster;

            $subset = $training->randomSubset($p);

            $booster->train($subset);

            $this->ensemble[] = $booster;

            /** @var list<int|float> $predictions */
            $predictions = $booster->predict($training);

            $out = Vector::quick($predictions)
                ->multiply($this->rate)
                ->add($prevOut);

            $loss = $gradient->square()->mean();

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->info('Numerical instability detected');
                }

                break;
            }

            $this->steps[] = $loss;

            if (isset($prevPred)) {
                /** @var list<int|float> $predictions */
                $predictions = $booster->predict($testing);

                $pred = Vector::quick($predictions)
                    ->multiply($this->rate)
                    ->add($prevPred);

                $score = $this->metric->score($pred->asArray(), $testing->labels());

                $this->scores[] = $score;
            }

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - {$this->metric}: "
                    . ($score ?? 'n/a') . ", L2 Loss: $loss");
            }

            if (isset($pred)) {
                if ($score >= $max) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestEpoch = $epoch;

                    $delta = 0;
                } else {
                    ++$delta;
                }

                if ($delta >= $this->window) {
                    break;
                }

                $prevPred = $pred;
            }

            if ($loss <= 0.0) {
                break;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break;
            }

            $prevOut = $out;
            $prevLoss = $loss;
        }

        if ($this->scores and end($this->scores) < $bestScore) {
            if ($this->logger) {
                $this->logger->info("Restoring ensemble state to epoch $bestEpoch");
            }

            $this->ensemble = array_slice($this->ensemble, 0, $bestEpoch);
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make a prediction from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<int|float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->ensemble or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        /** @var list<int|float> $predictions */
        $predictions = $this->base->predict($dataset);

        foreach ($this->ensemble as $estimator) {
            /** @var int $j */
            foreach ($estimator->predict($dataset) as $j => $prediction) {
                $predictions[$j] += $this->rate * $prediction;
            }
        }

        return $predictions;
    }

    /**
     * Return the normalized importance scores of each feature column of the training set.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if (!$this->ensemble or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $importances = array_fill(0, $this->featureCount, 0.0);

        foreach ($this->ensemble as $tree) {
            foreach ($tree->featureImportances() as $column => $importance) {
                $importances[$column] += $importance;
            }
        }

        $n = count($this->ensemble);

        foreach ($importances as &$importance) {
            $importance /= $n;
        }

        return $importances;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Gradient Boost (' . Params::stringify($this->params()) . ')';
    }
}
