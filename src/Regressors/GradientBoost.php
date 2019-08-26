<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\Tensor\Vector;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Other\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Gradient Boost
 *
 * Gradient Boost is a stage-wise additive ensemble that uses a Gradient
 * Descent boosting paradigm for training *weak* regressors (using
 * Regression Trees) to correct the error residuals of a base learner.
 *
 * > **Note**: The default base classifier is a Dummy Classifier using the
 * *Mean* Strategy and the default booster is a Regression Tree with a max
 * depth of 3.
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
class GradientBoost implements Estimator, Learner, Verbose, Persistable
{
    use PredictsSingle, LoggerAware;

    public const COMPATIBLE_BOOSTERS = [
        RegressionTree::class,
        ExtraTreeRegressor::class,
    ];

    /**
     * The weak regressor that will fix up the error residuals of the base
     * learner.
     *
     * @var \Rubix\ML\Learner
     */
    protected $booster;

    /**
     *  The max number of estimators to train in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The learning rate i.e the step size.
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
     * The minimum change in the cost function necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The holdout of training samples to use for validation. i.e. the holdout
     * holdout.
     *
     * @var float
     */
    protected $holdout;

    /**
     * The number of epochs without improvement in the validation score to wait
     * before considering an early stop.
     *
     * @var int
     */
    protected $window;

    /**
     * The metric used to score the generalization performance of the model
     * during training.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Metric
     */
    protected $metric;

    /**
     * The base regressor to be boosted.
     *
     * @var \Rubix\ML\Learner
     */
    protected $base;

    /**
     * An ensemble of weak regressors.
     *
     * @var array
     */
    protected $ensemble = [
        //
    ];

    /**
     * The number of feature columns in the training set.
     *
     * @var int|null
     */
    protected $featureCount;

    /**
     * The validation scores at each epoch.
     *
     * @var array
     */
    protected $scores = [
        //
    ];

    /**
     * The average cost of a training sample at each epoch.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param \Rubix\ML\Learner|null $booster
     * @param float $rate
     * @param int $estimators
     * @param float $ratio
     * @param float $minChange
     * @param float $holdout
     * @param int $window
     * @param \Rubix\ML\CrossValidation\Metrics\Metric|null $metric
     * @param \Rubix\ML\Learner|null $base
     * @throws \InvalidArgumentException
     */
    public function __construct(
        ?Learner $booster = null,
        float $rate = 0.1,
        int $estimators = 1000,
        float $ratio = 0.5,
        float $minChange = 1e-4,
        float $holdout = 0.1,
        int $window = 10,
        ?Metric $metric = null,
        ?Learner $base = null
    ) {
        if ($booster and !in_array(get_class($booster), self::COMPATIBLE_BOOSTERS)) {
            throw new InvalidArgumentException('Booster learner is not'
                . ' compatible with the ensemble.');
        }

        if ($rate <= 0. or $rate > 1.) {
            throw new InvalidArgumentException('Learning rate must be between'
                . " 0 and 1, $rate given.");
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . " 1 estimator, $estimators given.");
        }

        if ($ratio <= 0. or $ratio > 1.) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0 and 1, $ratio given.");
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if ($holdout < 0.01 or $holdout > 0.5) {
            throw new InvalidArgumentException('Holdout ratio must be between'
                . " 0.01 and 0.5, $holdout given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be at least 1'
                . " epoch, $window given.");
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::check($this, $metric);
        }

        if ($base and $base->type() !== self::REGRESSOR) {
            throw new InvalidArgumentException('Base estimator must be a'
                . ' regressor, ' . self::TYPES[$base->type()] . ' given.');
        }

        $this->booster = $booster ?? new RegressionTree(3);
        $this->rate = $rate;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->minChange = $minChange;
        $this->holdout = $holdout;
        $this->window = $window;
        $this->metric = $metric ?? new RSquared();
        $this->base = $base ?? new DummyRegressor(new Mean());
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
        $compatibility = array_intersect(
            $this->base->compatibility(),
            $this->booster->compatibility()
        );

        return array_values($compatibility);
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
     * Return the validation scores at each epoch.
     *
     * @return array
     */
    public function scores() : array
    {
        return $this->scores;
    }

    /**
     * Return the average cost at every epoch.
     *
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Train the estimator with a dataset.
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

        if ($this->logger) {
            $this->logger->info('Learner init ' . Params::stringify([
                'booster' => $this->booster,
                'rate' => $this->rate,
                'estimators' => $this->estimators,
                'ratio' => $this->ratio,
                'min_change' => $this->minChange,
                'hold_out' => $this->holdout,
                'window' => $this->window,
                'metric' => $this->metric,
                'base' => $this->base,
            ]));
        }

        $this->featureCount = $dataset->numColumns();

        [$testing, $training] = $dataset->randomize()->split($this->holdout);

        [$min, $max] = $this->metric->range();

        if ($this->logger) {
            $this->logger->info('Training base learner');
        }

        $this->base->train($training);

        $this->ensemble = $this->scores = $this->steps = [];

        $out = $prevOut = Vector::quick($this->base->predict($training));
        $target = Vector::quick($training->labels());

        $prevPred = Vector::quick($this->base->predict($testing));

        $k = (int) round($this->ratio * $training->numRows());

        $bestScore = $min;
        $bestEpoch = $nu = 0;
        $prevLoss = INF;

        for ($epoch = 1; $epoch <= $this->estimators; $epoch++) {
            $gradient = $target->subtract($out);
    
            $training = Labeled::quick($training->samples(), $gradient->asArray());

            $booster = clone $this->booster;

            $subset = $training->randomSubset($k);

            $booster->train($subset);

            $out = Vector::quick($booster->predict($training))
                ->multiply($this->rate)
                ->add($prevOut);

            $loss = $gradient->square()->mean();

            $pred = Vector::quick($booster->predict($testing))
                ->multiply($this->rate)
                ->add($prevPred);

            $score = $this->metric->score($pred->asArray(), $testing->labels());

            $this->ensemble[] = $booster;
            $this->steps[] = $loss;
            $this->scores[] = $score;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch score=$score loss=$loss");
            }

            if ($score > $bestScore) {
                $bestScore = $score;
                $bestEpoch = $epoch;

                $nu = 0;
            } else {
                $nu++;
            }

            if (is_nan($loss) or is_nan($score)) {
                break 1;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break 1;
            }

            if ($loss < EPSILON or $score >= $max) {
                break 1;
            }

            if ($nu >= $this->window) {
                break 1;
            }

            $prevOut = $out;
            $prevPred = $pred;
            $prevLoss = $loss;
        }

        if (end($this->scores) < $bestScore) {
            $this->ensemble = array_slice($this->ensemble, 0, $bestEpoch);

            if ($this->logger) {
                $this->logger->info("Ensemble restored to epoch $bestEpoch");
            }
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make a prediction from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->base->trained() or empty($this->ensemble)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = $this->base->predict($dataset);

        foreach ($this->ensemble as $estimator) {
            foreach ($estimator->predict($dataset) as $j => $prediction) {
                $predictions[$j] += $this->rate * $prediction;
            }
        }

        return $predictions;
    }

    /**
     * Return the normalized feature importances i.e. the proportion that each
     * feature contributes to the overall model, indexed by feature column.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function featureImportances() : array
    {
        if (!$this->ensemble or !$this->featureCount) {
            throw new RuntimeException('The estimator has not been trained.');
        }

        $importances = array_fill(0, $this->featureCount, 0.);

        foreach ($this->ensemble as $tree) {
            foreach ($tree->featureImportances() as $column => $value) {
                $importances[$column] += $value;
            }
        }

        $n = count($this->ensemble);

        foreach ($importances as &$importance) {
            $importance /= $n;
        }

        return $importances;
    }
}
