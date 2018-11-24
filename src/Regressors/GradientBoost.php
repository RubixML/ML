<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Ensemble;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Traits\LoggerAware;
use InvalidArgumentException;
use RuntimeException;

/**
 * Gradient Boost
 *
 * Gradient Boost is a stage-wise additive ensemble that uses a Gradient Descent
 * boosting paradigm for training *weak* regressors (usually Regression Trees) to
 * correct the error residuals of a base learner.
 *
 * References:
 * [1] J. H. Friedman. (2001). Greedy Function Approximation: A Gradient
 * Boosting Machine.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GradientBoost implements Learner, Ensemble, Verbose, Persistable
{
    use LoggerAware;

    const AVAILABLE_ESTIMATORS = [
        RegressionTree::class,
        ExtraTreeRegressor::class,
    ];

    /**
     *  The base regressor to be boosted.
     * 
     * @var \Rubix\ML\Learner
     */
    protected $base;

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
     * @param  \Rubix\ML\Learner|null  $booster
     * @param  int  $estimators
     * @param  float  $rate
     * @param  float  $ratio
     * @param  float  $tolerance
     * @param  \Rubix\ML\Learner|null  $base
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(?Learner $booster = null, int $estimators = 100, float $rate = 0.1,
                                float $ratio = 0.8, float $tolerance = 1e-4, ?Learner $base = null)
    {
        if (is_null($booster)) {
            $booster = new RegressionTree(3);
        }

        if (!in_array(get_class($booster), self::AVAILABLE_ESTIMATORS)) {
            throw new InvalidArgumentException('The estimator chosen as the'
                . ' booster is not compatible with gradient boost.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . " 1 estimator, $estimators given.");
        }

        if ($rate < 0.) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . " than 0, $rate given.");
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException('Subsample ratio must be between'
                . " 0.01 and 1, $ratio given.");
        }

        if ($tolerance < 0.) {
            throw new InvalidArgumentException('Tolerance cannot be less than'
                . " 0, $tolerance given.");
        }

        if (is_null($base)) {
            $base = new DummyRegressor(new Mean());
        }

        if ($base->type() !== self::REGRESSOR) {
            throw new InvalidArgumentException('Base estimator must be a'
                . ' regressor, ' . self::TYPES[$base->type()] . ' given.');
        }

        $this->booster = $booster;
        $this->estimators = $estimators;
        $this->rate = $rate;
        $this->ratio = $ratio;
        $this->tolerance = $tolerance;
        $this->base = $base;
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
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        if ($this->logger) $this->logger->info('Learner initialized w/ '
            . Params::stringify([
                'base' => $this->base,
                'booster' => $this->booster,
                'estimators' => $this->estimators,
                'rate' => $this->rate,
                'ratio' => $this->ratio,
                'tolerance' => $this->tolerance,
            ]));

        $n = $dataset->numRows();
        $p = (int) round($this->ratio * $n);

        if ($this->logger) $this->logger->info('Training '
            . Params::shortName($this->base) . ' base estimator');

        $this->base->train($dataset);

        $predictions = $this->base->predict($dataset);

        $yHat = [];

        foreach ($predictions as $i => $prediction) {
            $yHat[] = $dataset->label($i) - $prediction;
        }

        $residuals = Labeled::quick($dataset->samples(), $yHat);

        if ($this->logger) $this->logger->info('Attempting to correct residuals'
            . " w/ $this->estimators " . Params::shortName($this->booster)
            . ($this->estimators > 1 ? 's' : ''));

        $this->ensemble = $this->steps = [];

        $previous = INF;

        for ($epoch = 1; $epoch <= $this->estimators; $epoch++) {
            $booster = clone $this->booster;

            $subset = $residuals->randomize()->head($p);

            $booster->train($subset);

            $predictions = $booster->predict($residuals);

            $loss = 0.;
            $yHat = [];

            foreach ($predictions as $i => $prediction) {
                $label = $residuals->label($i);

                $loss += ($label - $prediction) ** 2;
                $yHat[] = $label - ($this->rate * $prediction);
            }

            $loss /= $n;

            $this->ensemble[] = $booster;
            $this->steps[] = $loss;

            if ($this->logger) $this->logger->info("Epoch $epoch"
                . " complete, loss=$loss");

            if ($loss < $this->tolerance) {
                break 1;
            }

            $residuals = Labeled::quick($residuals->samples(), $yHat);

            $previous = $loss;
        }

        if ($this->logger) $this->logger->info('Training complete');
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

        $predictions = $this->base->predict($dataset);

        foreach ($this->ensemble as $estimator) {
            foreach ($estimator->predict($dataset) as $j => $prediction) {
                $predictions[$j] += $this->rate * $prediction;
            }
        }

        return $predictions;
    }
}