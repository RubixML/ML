<?php

namespace Rubix\ML;

use Amp\Loop;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Amp\Parallel\Worker\DefaultPool;
use Amp\Parallel\Worker\CallableTask;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;

use function Amp\call;
use function Amp\Promise\all;

/**
 * Committee Machine
 *
 * A voting ensemble that aggregates the predictions of a committee of heterogeneous
 * estimators (called *experts*). The committee uses a user-specified influence-based
 * scheme to sway final predictions.
 *
 * > **Note**: Influence values can be arbitrary as they are normalized upon object
 * creation.
 *
 * References:
 * [1] H. Drucker. (1997). Fast Committee Machines for Regression and Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CommitteeMachine implements Estimator, Learner, Parallel, Verbose, Persistable
{
    use Multiprocessing, LoggerAware;

    protected const COMPATIBLE_ESTIMATOR_TYPES = [
        self::CLASSIFIER,
        self::REGRESSOR,
        self::ANOMALY_DETECTOR,
    ];

    /**
     * The committee of experts. i.e. the ensemble of estimators.
     *
     * @var array
     */
    protected $experts;

    /**
     * The influence values of each expert in the committee.
     *
     * @var (int|float)[]
     */
    protected $influences;

    /**
     * The type of estimator this is.
     *
     * @var int
     */
    protected $type;

    /**
     * The data types that the committee is compatible with.
     *
     * @var int[]
     */
    protected $compatibility;

    /**
     * The possible class labels.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param array $experts
     * @param array|null $influences
     * @throws \InvalidArgumentException
     */
    public function __construct(array $experts, ?array $influences = null)
    {
        $p = count($experts);

        if ($p < 1) {
            throw new InvalidArgumentException('Committee must contain at least'
                . ' 1 expert, none given.');
        }

        foreach ($experts as $expert) {
            if (!$expert instanceof Learner) {
                throw new InvalidArgumentException('Expert must implement the'
                    . ' learner interface.');
            }
        }

        $prototype = reset($experts);

        $type = $prototype->type();

        if (!in_array($type, self::COMPATIBLE_ESTIMATOR_TYPES)) {
            throw new InvalidArgumentException('This meta estimator'
                . ' only supports classifiers, regressors, and anomaly'
                . ' detectors, ' . self::TYPES[$type] . ' given.');
        }

        foreach ($experts as $expert) {
            if ($expert->type() !== $type) {
                throw new InvalidArgumentException('Experts must be of the'
                    . ' same type, ' . self::TYPES[$type] . ' expected but'
                    . ' found ' . self::TYPES[$expert->type()] . '.');
            }
        }

        if (is_array($influences)) {
            if (count($influences) !== $p) {
                throw new InvalidArgumentException('The number of influence'
                    . " values must equal the number of experts, $p needed"
                    . ' but ' . count($influences) . 'given.');
            }

            $total = array_sum($influences);

            if ($total == 0) {
                throw new InvalidArgumentException('Total influence for the'
                    . ' committee cannot be 0.');
            }

            foreach ($influences as &$influence) {
                $influence /= $total;
            }
        } else {
            $influences = array_fill(0, $p, 1 / $p);
        }

        $compatibility = array_intersect(...array_map(function ($estimator) {
            return $estimator->compatibility();
        }, $experts));

        if (count($compatibility) < 1) {
            throw new InvalidArgumentException('Incompatible committee.');
        }

        $this->experts = $experts;
        $this->influences = $influences;
        $this->type = $type;
        $this->compatibility = array_values($compatibility);
        $this->workers = min(DEFAULT_WORKERS, $p);
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->type;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->compatibility;
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return reset($this->experts)->trained();
    }

    /**
     * Return the learner instances of the committee.
     *
     * @return \Rubix\ML\Learner[]
     */
    public function experts() : array
    {
        return $this->experts;
    }

    /**
     * Return the normalized influence values for each expert in the committee.
     *
     * @return array
     */
    public function influences() : array
    {
        return $this->influences;
    }

    /**
     * Train all the experts with the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if ($this->type === self::CLASSIFIER or $this->type === self::REGRESSOR) {
            if (!$dataset instanceof Labeled) {
                throw new InvalidArgumentException('This estimator requires a'
                    . ' labeled training set.');
            }
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner init ' . Params::stringify([
                'experts' => $this->experts,
                'influences' => $this->influences,
                'workers' => $this->workers,
            ]));
        }

        Loop::run(function () use ($dataset) {
            $pool = new DefaultPool($this->workers);

            $coroutines = [];

            foreach ($this->experts as $i => $estimator) {
                $task = new CallableTask(
                    [$this, '_train'],
                    [$estimator, $dataset]
                );

                $coroutines[] = call(function () use ($pool, $task, $i) {
                    $estimator = yield $pool->enqueue($task);

                    if ($this->logger) {
                        $this->logger->info(Params::stringify([
                            $i => $estimator,
                        ]) . ' finished');
                    }

                    return $estimator;
                });
            }

            $this->experts = yield all($coroutines);
            
            return yield $pool->shutdown();
        });

        if ($this->type === self::CLASSIFIER and $dataset instanceof Labeled) {
            $this->classes = $dataset->possibleOutcomes();
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $aggregate = [];

        foreach ($this->experts as $estimator) {
            foreach ($estimator->predict($dataset) as $i => $prediction) {
                $aggregate[$i][] = $prediction;
            }
        }

        switch ($this->type) {
            case self::CLASSIFIER:
                return array_map([$this, 'decideClass'], $aggregate);

            case self::REGRESSOR:
                return array_map([$this, 'decideValue'], $aggregate);

            case self::ANOMALY_DETECTOR:
                return array_map([$this, 'decideAnomaly'], $aggregate);
        }
    }

    /**
     * Decide on a class outcome.
     *
     * @param (int|string)[] $votes
     * @return int|string
     */
    public function decideClass(array $votes)
    {
        $scores = array_fill_keys($this->classes, 0.);

        foreach ($votes as $i => $vote) {
            $scores[$vote] += $this->influences[$i];
        }

        return argmax($scores);
    }

    /**
     * Decide on a real valued outcome.
     *
     * @param (int|float)[] $votes
     * @return float
     */
    public function decideValue(array $votes) : float
    {
        return Stats::weightedMean($votes, $this->influences);
    }

    /**
     * Decide on an anomaly outcome.
     *
     * @param int[] $votes
     * @return int
     */
    public function decideAnomaly(array $votes) : int
    {
        $scores = array_fill(0, 2, 0.);

        foreach ($votes as $i => $vote) {
            $scores[$vote] += $this->influences[$i];
        }

        return argmax($scores);
    }

    /**
     * Train an learner using a dataset and return it.
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
}
