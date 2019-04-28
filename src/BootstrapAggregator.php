<?php

namespace Rubix\ML;

use Amp\Loop;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Amp\Parallel\Worker\DefaultPool;
use Amp\Parallel\Worker\CallableTask;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Amp\call;
use function Amp\Promise\all;

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
     * The max number of processes to run in parallel for training.
     *
     * @var int
     */
    protected $workers;

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
     * @param int $workers
     * @throws \InvalidArgumentException
     */
    public function __construct(Learner $base, int $estimators = 10, float $ratio = 0.5, int $workers = 4)
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
        
        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0.01 and 1, $ratio given.");
        }

        if ($workers < 1) {
            throw new InvalidArgumentException('Cannot have less than'
                . " 1 worker process, $workers given.");
        }

        $this->base = $base;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->workers = $workers;
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
        if ($this->type() === self::CLASSIFIER or $this->type() === self::REGRESSOR) {
            if (!$dataset instanceof Labeled) {
                throw new InvalidArgumentException('This estimator requires a'
                    . ' labeled training set.');
            }
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $p = (int) round($this->ratio * $dataset->numRows());

        $this->ensemble = [];

        Loop::run(function () use ($dataset, $p) {
            $pool = new DefaultPool($this->workers);

            $coroutines = [];

            for ($i = 0; $i < $this->estimators; $i++) {
                $estimator = clone $this->base;

                $subset = $dataset->randomSubsetWithReplacement($p);

                $task = new CallableTask(
                    [$this, '_train'],
                    [$estimator, $subset]
                );

                $coroutines[] = call(function () use ($pool, $task) {
                    return yield $pool->enqueue($task);
                });
            }

            $this->ensemble = yield all($coroutines);
            
            return yield $pool->shutdown();
        });
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
                return array_map([self::class, 'decideClass'], $aggregate);

            case self::ANOMALY_DETECTOR:
                return array_map([self::class, 'decideAnomaly'], $aggregate);

            default:
                return array_map([Stats::class, 'mean'], $aggregate);

        }
    }

    /**
     * Train an estimator using a dataset and return it.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Learner
     */
    public function _train(Learner $estimator, Dataset $dataset) : Learner
    {
        $estimator->train($dataset);

        return $estimator;
    }

    /**
     * Classification decision function.
     *
     * @param (int|string)[] $outcomes
     * @return int|string
     */
    public function decideClass($outcomes)
    {
        return argmax(array_count_values($outcomes));
    }

    /**
     * Anomaly detection decision function.
     *
     * @param int[] $outcomes
     * @return int
     */
    public function decideAnomaly($outcomes) : int
    {
        return Stats::mean($outcomes) > 0.5 ? 1 : 0;
    }
}
