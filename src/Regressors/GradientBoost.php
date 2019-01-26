<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

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
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GradientBoost implements Learner, Verbose, Persistable
{
    use LoggerAware;

    const AVAILABLE_BOOSTERS = [
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
     * The amount of mean squared error to tolerate before early stopping.
     *
     * @var float
     */
    protected $tolerance;

    /**
     * The base regressor to be boosted.
     * 
     * @var \Rubix\ML\Learner
     */
    protected $base;

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
     * @param  float  $rate
     * @param  int  $estimators
     * @param  float  $ratio
     * @param  float  $minChange
     * @param  float  $tolerance
     * @param  \Rubix\ML\Learner|null  $base
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(?Learner $booster = null, float $rate = 0.1, int $estimators = 100,
                                float $ratio = 0.8, float $minChange = 1e-4, float $tolerance = 1e-3,
                                ?Learner $base = null)
    {
        if (is_null($booster)) {
            $booster = new RegressionTree(3);
        }

        if (!in_array(get_class($booster), self::AVAILABLE_BOOSTERS)) {
            throw new InvalidArgumentException('The estimator chosen as the'
                . ' booster is not compatible with gradient boost.');
        }

        if ($rate < 0.) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . " than 0, $rate given.");
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . " 1 estimator, $estimators given.");
        }

        if ($ratio < 0.01 or $ratio > 0.99) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0.01 and 0.99, $ratio given.");
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . " than 0, $minChange given.");
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
        $this->rate = $rate;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->minChange = $minChange;
        $this->tolerance = $tolerance;
        $this->base = $base;
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
        return array_values(array_intersect($this->base->compatibility(), $this->booster->compatibility()));
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

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) $this->logger->info('Learner initialized w/ '
            . Params::stringify([
                'booster' => $this->booster,
                'rate' => $this->rate,
                'estimators' => $this->estimators,
                'ratio' => $this->ratio,
                'min_change' => $this->minChange,
                'tolerance' => $this->tolerance,
                'base' => $this->base,
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

            if (is_nan($loss)) {
                break 1;
            }

            if (abs($previous - $loss) < $this->minChange) {
                break 1;
            }

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
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->ensemble)) {
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
}