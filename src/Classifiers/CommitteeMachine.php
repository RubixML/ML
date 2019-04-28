<?php

namespace Rubix\ML\Classifiers;

use Amp\Loop;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Amp\Parallel\Worker\DefaultPool;
use Amp\Parallel\Worker\CallableTask;
use InvalidArgumentException;

use function Amp\call;
use function Amp\Promise\all;

/**
 * Committee Machine
 *
 * A voting ensemble that aggregates the predictions of a committee of heterogeneous
 * classifiers (called *experts*). The committee uses a user-specified influence-based
 * scheme to sway final predictions.
 *
 * > **Note**: Influence values can be arbitrary as they are normalized upon object
 * creation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CommitteeMachine implements Estimator, Learner, Probabilistic, Persistable
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
     * The data types that the committee is compatible with.
     *
     * @var int[]
     */
    protected $compatibility;

    /**
     * The unique class labels.
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
        $k = count($experts);

        if ($k < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . ' 1 estimator, 0 given.');
        }

        foreach ($experts as $expert) {
            if (!$expert instanceof Learner) {
                throw new InvalidArgumentException('Base estimator must'
                    . ' implement the learner interface.');
            }

            if ($expert->type() !== self::CLASSIFIER) {
                throw new InvalidArgumentException('Base estimator must be a'
                    . ' classifier, ' . self::TYPES[$expert->type()] . ' given.');
            }

            if (!$expert instanceof Probabilistic) {
                throw new InvalidArgumentException('Base estimator must'
                    . ' implement the probabilistic interface.');
            }
        }

        if (is_array($influences)) {
            if (count($influences) !== $k) {
                throw new InvalidArgumentException('The number of influence'
                    . " values must equal the number of experts, $k needed"
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
            $influences = array_fill(0, $k, 1 / $k);
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
        $this->compatibility = $compatibility;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
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
        return !empty($this->classes);
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
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

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

        $this->classes = $dataset->possibleOutcomes();
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map('Rubix\ML\argmax', $this->proba($dataset));
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $probabilities = array_fill(
            0,
            $dataset->numRows(),
            array_fill_keys($this->classes, 0.)
        );

        foreach ($this->experts as $i => $expert) {
            $influence = $this->influences[$i];

            foreach ($expert->proba($dataset) as $j => $joint) {
                foreach ($joint as $class => $proba) {
                    $probabilities[$j][$class] += $influence * $proba;
                }
            }
        }

        return $probabilities;
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
}
