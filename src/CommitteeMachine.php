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
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CommitteeMachine implements Estimator, Learner, Persistable
{
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
     * The max number of processes to run in parallel for training.
     *
     * @var int
     */
    protected $workers;

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
     * @param int $workers
     * @throws \InvalidArgumentException
     */
    public function __construct(array $experts, ?array $influences = null, int $workers = 4)
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

        if (
            $type !== self::CLASSIFIER and
            $type !== self::REGRESSOR and
            $type !== self::ANOMALY_DETECTOR
        ) {
            throw new InvalidArgumentException('Expert must be a classifier,'
                . ' regressor, or anomaly detector.');
        }

        foreach ($experts as $expert) {
            if ($expert->type() !== $type) {
                throw new InvalidArgumentException('Experts must be of the'
                 . ' same type.');
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

        if ($workers < 1) {
            throw new InvalidArgumentException('Cannot have less than'
                . " 1 worker process, $workers given.");
        }

        $compatibility = array_intersect(...array_map(function ($estimator) {
            return $estimator->compatibility();
        }, $experts));

        if (count($compatibility) < 1) {
            throw new InvalidArgumentException('Committee must only'
                . ' contain estimators that share at least 1 data type'
                . ' they are compatible with.');
        }

        $this->experts = $experts;
        $this->influences = $influences;
        $this->workers = $workers;
        $this->type = $type;
        $this->compatibility = array_values($compatibility);
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

        Loop::run(function () use ($dataset) {
            $pool = new DefaultPool($this->workers);

            $coroutines = [];

            foreach ($this->experts as $estimator) {
                $task = new CallableTask(
                    [$this, '_train'],
                    [$estimator, $dataset]
                );

                $coroutines[] = call(function () use ($pool, $task) {
                    return yield $pool->enqueue($task);
                });
            }

            $this->experts = yield all($coroutines);
            
            return yield $pool->shutdown();
        });

        if ($this->type === self::CLASSIFIER and $dataset instanceof Labeled) {
            $this->classes = $dataset->possibleOutcomes();
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
        $votes = [];

        foreach ($this->experts as $expert) {
            foreach ($expert->predict($dataset) as $i => $prediction) {
                $votes[$i][] = $prediction;
            }
        }

        switch ($this->type) {
            case self::CLASSIFIER:
                return array_map([$this, 'decideClass'], $votes);

            case self::REGRESSOR:
                return array_map([$this, 'decideValue'], $votes);

            case self::ANOMALY_DETECTOR:
                return array_map([$this, 'decideAnomaly'], $votes);
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
