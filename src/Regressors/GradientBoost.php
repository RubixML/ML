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
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GradientBoost implements Estimator, Learner, Verbose, Persistable
{
    use LoggerAware;

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
     * @param \Rubix\ML\Learner|null $booster
     * @param float $rate
     * @param int $estimators
     * @param float $ratio
     * @param float $minChange
     * @param \Rubix\ML\Learner|null $base
     * @throws \InvalidArgumentException
     */
    public function __construct(
        ?Learner $booster = null,
        float $rate = 0.1,
        int $estimators = 100,
        float $ratio = 1.0,
        float $minChange = 1e-4,
        ?Learner $base = null
    ) {
        if ($booster and !in_array(get_class($booster), self::COMPATIBLE_BOOSTERS)) {
            throw new InvalidArgumentException('Booster learner is not'
                . ' compatible with ensemble.');
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

        if ($base and $base->type() !== self::REGRESSOR) {
            throw new InvalidArgumentException('Base estimator must be a'
                . ' regressor, ' . self::TYPES[$base->type()] . ' given.');
        }

        $this->booster = $booster ?? new RegressionTree(3);
        $this->rate = $rate;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->minChange = $minChange;
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
            throw new InvalidArgumentException('This estimator requires a'
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
                'base' => $this->base,
            ]));
        }

        $p = (int) round($this->ratio * $dataset->numRows());

        if ($this->logger) {
            $this->logger->info('Training base estimator');
        }

        $this->base->train($dataset);

        $target = Vector::quick($dataset->labels());
        $output = Vector::quick($this->base->predict($dataset));

        $gradient = $this->gradient($output, $target)->asArray();
    
        $dataset = Labeled::quick($dataset->samples(), $gradient);

        $this->ensemble = $this->steps = [];

        $prevOut = $output;
        $previous = INF;

        for ($epoch = 1; $epoch <= $this->estimators; $epoch++) {
            $booster = clone $this->booster;

            $subset = $dataset->randomSubsetWithoutReplacement($p);

            $booster->train($subset);

            $predictions = $booster->predict($dataset);

            $output = Vector::quick($predictions)
                ->multiply($this->rate)
                ->add($prevOut);

            $loss = $this->loss($output, $target);

            $this->ensemble[] = $booster;
            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch loss=$loss");
            }

            if (is_nan($loss) or $loss < EPSILON) {
                break 1;
            }

            if (abs($previous - $loss) < $this->minChange) {
                break 1;
            }

            $gradient = $this->gradient($output, $target)->asArray();

            $dataset = Labeled::quick($dataset->samples(), $gradient);

            $previous = $loss;
            $prevOut = $output;
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

    /**
     * Calculate the mean squared error training loss.
     *
     * @param \Rubix\Tensor\Vector $output
     * @param \Rubix\Tensor\Vector $target
     * @return float
     */
    public function loss(Vector $output, Vector $target) : float
    {
        return $target->subtract($output)->square()->mean();
    }

    /**
     * Compute the negative gradient of the cost function.
     *
     * @param \Rubix\Tensor\Vector $output
     * @param \Rubix\Tensor\Vector $target
     * @return \Rubix\Tensor\Vector
     */
    public function gradient(Vector $output, Vector $target) : Vector
    {
        return $target->subtract($output);
    }
}
