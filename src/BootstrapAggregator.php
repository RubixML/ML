<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Functions\Argmax;
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
class BootstrapAggregator implements Learner, Persistable
{
    const COMPATIBLE_ESTIMATOR_TYPES = [
        self::CLASSIFIER,
        self::REGRESSOR,
        self::DETECTOR,
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
     * @var array
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
                . ' detectors, ' . self::TYPES[$base->type()] . ' given.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must train at'
                . " least 1 estimator, $estimators given.");
        }

        if ($ratio < 0.01 or $ratio > 0.99) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0.01 and 0.99, $ratio given.");
        }

        $this->base = $base;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
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
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
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
        $p = (int) round($this->ratio * $dataset->numRows());

        $this->ensemble = [];

        for ($epoch = 1; $epoch <= $this->estimators; $epoch++) {
            $estimator = clone $this->base;

            $subset = $dataset->randomSubsetWithReplacement($p);

            $estimator->train($subset);

            $this->ensemble[] = $estimator;
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->ensemble)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $aggregate = [];

        foreach ($this->ensemble as $estimator) {
            foreach ($estimator->predict($dataset) as $i => $prediction) {
                $aggregate[$i][] = $prediction;
            }
        }
        
        switch ($this->type()) {
            case self::CLASSIFIER:
                return array_map(function ($outcomes) {
                    return Argmax::compute(array_count_values($outcomes));
                }, $aggregate);

            case self::DETECTOR:
                return array_map(function ($outcomes) {
                    return Stats::mean($outcomes) > 0.5 ? 1 : 0;
                }, $aggregate);

            default:
                return array_map([Stats::class, 'mean'], $aggregate);

        }
    }
}
