<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;
use RuntimeException;

/**
 * Gradient Boost
 *
 * Gradient Boost is a stage-wise additive ensemble that uses a Gradient
 * Descent boosting paradigm for training the base "weak" regressors.
 * Sepcifically, gradient boosting attempts to improve bias by training
 * subsequent estimators to correct for errors made by the previous
 * learners.
 *
 * References:
 * [1] J. H. Friedman. (2001). Greedy Function Approximation: A Gradient
 * Boosting Machine.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GradientBoost implements Estimator, Ensemble, Persistable
{
    const AVAILABLE_ESTIMATORS = [
        RegressionTree::class,
        ExtraTreeRegressor::class,
    ];

    /**
     *  The base regressor to be boosted.
     * 
     * @var \Rubix\ML\Estimator
     */
    protected $base;

    /**
     *  The max number of estimators to train in the ensemble.
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
     *  The minimum change in the loss to continue training.
     * 
     * @var float
     */
    protected $minChange;

    /**
     * The amount of mean squared error to tolerate before early
     * stopping.
     *
     * @var float
     */
    protected $tolerance;

    /**
     * The ensemble of "weak" regressors.
     *
     * @var array
     */
    protected $ensemble = [
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
     * @param  \Rubix\ML\Estimator  $base
     * @param  int  $estimators
     * @param  float  $rate
     * @param  float  $ratio
     * @param  float  $tolerance
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(Estimator $base = null, int $estimators = 100, float $rate = 0.1,
                            float $ratio = 0.8, float $minChange = 1e-4, float $tolerance = 1e-5)
    {
        if (is_null($base)) {
            $base = new RegressionTree(2);
        }

        if (!in_array(get_class($base), self::AVAILABLE_ESTIMATORS)) {
            throw new InvalidArgumentException('Base estimator is not'
                . ' compatible with gradient boost.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . ' 1 estimator.');
        }

        if ($rate < 0.) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . ' than 0.');
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException('Sample ratio must be between'
                . ' 0.01 and 1.0.');
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException('Minimum change must be greater'
                . ' than 0.');
        }

        if ($tolerance < 0. or $tolerance > 1.) {
            throw new InvalidArgumentException('Error tolerance must be between'
                . ' 0 and 1.');
        }

        $this->base = $base;
        $this->estimators = $estimators;
        $this->rate = $rate;
        $this->ratio = $ratio;
        $this->minChange = $minChange;
        $this->tolerance = $tolerance;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * Return the ensemble of estimators.
     *
     * @return array
     */
    public function estimators() : array
    {
        return $this->ensemble;
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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $n = $dataset->numRows();
        $p = (int) round($this->ratio * $n);

        $this->ensemble = $this->steps = [];

        $previous = INF;

        for ($epoch = 0; $epoch < $this->estimators; $epoch++) {
            $estimator = clone $this->base;

            $subset = $dataset->randomize()->head($p);

            $estimator->train($subset);

            $predictions = $estimator->predict($dataset);

            $loss = 0.;
            $yHat = [];

            foreach ($predictions as $i => $prediction) {
                $label = $dataset->label($i);

                $loss += ($label - $prediction) ** 2;
                $yHat[] = $label - ($this->rate * $prediction);
            }

            $loss /= $n;

            $this->ensemble[] = $estimator;
            $this->steps[] = $loss;

            if (abs($previous - $loss) < $this->minChange) {
                break 1;
            }

            if ($loss < $this->tolerance) {
                break 1;
            }

            $dataset = new Labeled($dataset->samples(), $yHat, false);

            $previous = $loss;
        }
    }

    /**
     * Make a prediction from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->ensemble)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = array_fill(0, $dataset->numRows(), 0.);

        foreach ($this->ensemble as $i => $estimator) {
            foreach ($estimator->predict($dataset) as $j => $prediction) {
                $predictions[$j] += $this->rate * $prediction;
            }
        }

        return $predictions;
    }
}