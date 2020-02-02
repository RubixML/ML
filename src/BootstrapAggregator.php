<?php

namespace Rubix\ML;

use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Bootstrap Aggregator
 *
 * Bootstrap Aggregating (or *bagging* for short) is a model averaging
 * technique designed to improve the stability and performance of a
 * user-specified base estimator by training a number of them on a unique
 * *bootstrapped* training set sampled at random with replacement.
 *
 * References:
 * [1] L. Breiman. (1996). Bagging Predictors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BootstrapAggregator implements Estimator, Learner, Parallel, Persistable
{
    use Multiprocessing, PredictsSingle;

    /**
     * The estimator types that this ensemble is compatible with.
     *
     * @var int[]
     */
    protected const COMPATIBLE_ESTIMATOR_TYPES = [
        self::CLASSIFIER,
        self::REGRESSOR,
        self::ANOMALY_DETECTOR,
    ];

    /**
     * The base estimator instance.
     *
     * @var \Rubix\ML\Learner
     */
    protected $base;

    /**
     * The number of estimators to train.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The ratio of training samples to train each estimator on.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The ensemble of estimators.
     *
     * @var \Rubix\ML\Learner[]
     */
    protected $ensemble = [
        //
    ];

    /**
     * @param \Rubix\ML\Learner $base
     * @param int $estimators
     * @param float $ratio
     * @throws \InvalidArgumentException
     */
    public function __construct(Learner $base, int $estimators = 10, float $ratio = 0.5)
    {
        if (!in_array($base->type(), self::COMPATIBLE_ESTIMATOR_TYPES)) {
            throw new InvalidArgumentException('This meta estimator'
                . ' only supports classifiers, regressors, and anomaly'
                . ' detectors, ' . self::TYPE_STRINGS[$base->type()] . ' given.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must contain at'
                . " least 1 estimator, $estimators given.");
        }
        
        if ($ratio <= 0. or $ratio > 1.5) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0 and 1.5, $ratio given.");
        }

        $this->base = $base;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->backend = new Serial();
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->base->type();
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
            'estimators' => $this->estimators,
            'ratio' => $this->ratio,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !empty($this->ensemble);
    }

    /**
     * Instantiate and train each base estimator in the ensemble on a bootstrap
     * training set.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if ($this->type() === self::CLASSIFIER or $this->type() === self::REGRESSOR) {
            if (!$dataset instanceof Labeled) {
                throw new InvalidArgumentException('Learner requires a'
                    . ' labeled training set.');
            }
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $p = (int) round($this->ratio * $dataset->numRows());

        $this->backend->flush();

        for ($i = 0; $i < $this->estimators; ++$i) {
            $estimator = clone $this->base;

            $subset = $dataset->randomSubsetWithReplacement($p);

            $this->backend->enqueue(new Deferred(
                [self::class, '_train'],
                [$estimator, $subset]
            ));
        }

        $this->ensemble = $this->backend->process();
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->ensemble)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $this->backend->flush();

        foreach ($this->ensemble as $estimator) {
            $this->backend->enqueue(new Deferred(
                [self::class, '_predict'],
                [$estimator, $dataset]
            ));
        }

        $aggregate = array_transpose($this->backend->process());
        
        switch ($this->type()) {
            case self::CLASSIFIER:
                return array_map([self::class, 'decideDiscrete'], $aggregate);

            case self::REGRESSOR:
                return array_map([Stats::class, 'mean'], $aggregate);

            case self::ANOMALY_DETECTOR:
                return array_map([self::class, 'decideDiscrete'], $aggregate);

            default:
                throw new RuntimeException('Invalid estimator type.');
        }
    }

    /**
     * Decide on a discrete-valued outcome.
     *
     * @param string[] $votes
     * @return string
     */
    public function decideDiscrete($votes) : string
    {
        return argmax(array_count_values($votes));
    }

    /**
     * Train a single learner and return it.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Learner
     */
    public static function _train(Learner $estimator, Dataset $dataset) : Learner
    {
        $estimator->train($dataset);

        return $estimator;
    }

    /**
     * Return the predictions from an estimator.
     *
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return mixed[]
     */
    public static function _predict(Estimator $estimator, Dataset $dataset) : array
    {
        return $estimator->predict($dataset);
    }
}
