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
 * Bootstrap Aggregating (or bagging) is a model averaging technique designed to
 * improve the stability and performance of a user-specified base Estimator by
 * training a number of them on a unique bootstrapped training set. Bootstrap
 * Aggregator then collects all of their predictions and makes a final
 * prediction based on the results.
 * 
 * References:
 * [1] L. Breiman. (1996). Bagging Predictors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BootstrapAggregator implements MetaEstimator, Learner, Ensemble, Persistable
{
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
     * @param  \Rubix\ML\Learner  $base
     * @param  int  $estimators
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(Learner $base, int $estimators = 10, float $ratio = 0.5)
    {
        if ($base instanceof MetaEstimator) {
            throw new InvalidArgumentException('Base estimator cannot be a meta'
                . ' estimator.');
        }

        $type = $base->type();

        if ($type !== self::CLASSIFIER and $type !== self::REGRESSOR and $type !== self::DETECTOR) {
            throw new InvalidArgumentException('This meta estimator only'
                . ' supports classifiers, regressors, and anomaly detectors, '
                . self::TYPES[$base->type()] . ' given.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException("Ensemble must train at least"
                . " 1 estimator, $estimators given.");
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException("Subsample ratio must be between"
                . " 0.01 and 1, $ratio given.");
        }

        $this->base = $base;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->base->type();
    }

    /**
     * Return the base estimator instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function estimator() : Estimator
    {
        return $this->base;
    }

    /**
     * Instantiate and train each base estimator in the ensemble on a bootstrap
     * training set.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
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

        $type = $this->type();

        $predictions = [];

        foreach ($aggregate as $outcomes) {
            if ($type === self::CLASSIFIER) {
                $predictions[] = Argmax::compute(array_count_values($outcomes));
            } else if ($type === self::DETECTOR) {
                $predictions[] = Stats::mean($outcomes) > 0.5 ? 1 : 0;
            } else {
                $predictions[] = Stats::mean($outcomes);
            }
        }

        return $predictions;
    }
}
