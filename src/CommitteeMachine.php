<?php

namespace Rubix\ML;

use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function count;

/**
 * Committee Machine
 *
 * A voting ensemble that aggregates the predictions of a committee of heterogeneous
 * estimators (called *experts*). The committee uses a user-specified influence-based
 * scheme to weight the final predictions.
 *
 * > **Note**: Influence values can be arbitrary as they are normalized upon
 * instantiation.
 *
 * References:
 * [1] H. Drucker. (1997). Fast Committee Machines for Regression and Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CommitteeMachine implements Estimator, Learner, Parallel, Persistable, Verbose
{
    use Multiprocessing, PredictsSingle, LoggerAware;

    /**
     * The integer-encoded estimator types this ensemble is compatible with.
     *
     * @var int[]
     */
    protected const COMPATIBLE_ESTIMATOR_TYPES = [
        self::CLASSIFIER,
        self::REGRESSOR,
        self::ANOMALY_DETECTOR,
    ];

    /**
     * The committee of experts. i.e. the ensemble of estimators.
     *
     * @var \Rubix\ML\Learner[]
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
     * @var \Rubix\ML\DataType[]
     */
    protected $compatibility;

    /**
     * The possible class labels.
     *
     * @var string[]
     */
    protected $classes = [
        //
    ];

    /**
     * @param \Rubix\ML\Learner[] $experts
     * @param (int|float)[]|null $influences
     * @throws \InvalidArgumentException
     */
    public function __construct(array $experts, ?array $influences = null)
    {
        if (empty($experts)) {
            throw new InvalidArgumentException('Committee must contain at'
                . ' least 1 expert.');
        }

        $compatibilities = [];

        foreach ($experts as $expert) {
            if (!$expert instanceof Learner) {
                throw new InvalidArgumentException('Expert must implement'
                    . ' the learner interface.');
            }

            $compatibilities[] = $expert->compatibility();
        }

        $compatibility = array_values(array_intersect(...$compatibilities));

        if (count($compatibility) < 1) {
            throw new InvalidArgumentException('Committee must have at least'
                . ' 1 data type that they are compatible with in common.');
        }

        $type = reset($experts)->type();

        if (!in_array($type, self::COMPATIBLE_ESTIMATOR_TYPES)) {
            throw new InvalidArgumentException('This meta estimator'
                . ' only supports classifiers, regressors, and anomaly'
                . ' detectors, ' . self::TYPE_STRINGS[$type] . ' given.');
        }

        foreach ($experts as $expert) {
            if ($expert->type() !== $type) {
                throw new InvalidArgumentException('Experts must all be of'
                    . ' the same type, ' . self::TYPE_STRINGS[$type] . ' expected'
                    . ' but found ' . self::TYPE_STRINGS[$expert->type()] . '.');
            }
        }

        $k = count($experts);

        if ($influences) {
            if (count($influences) !== $k) {
                throw new InvalidArgumentException('The number of influence'
                    . " values must equal the number of experts, $k needed"
                    . ' but ' . count($influences) . ' given.');
            }

            $total = array_sum($influences);

            if ($total <= 0) {
                throw new InvalidArgumentException('Total influence must'
                    . " be greater than 0, $total given.");
            }

            foreach ($influences as &$weight) {
                $weight /= $total;
            }
        } else {
            $influences = array_fill(0, $k, 1.0 / $k);
        }

        $this->experts = $experts;
        $this->influences = $influences;
        $this->type = $type;
        $this->compatibility = $compatibility;
        $this->backend = new Serial();
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
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->compatibility;
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'experts' => $this->experts,
            'influences' => $this->influences,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->experts ? end($this->experts)->trained() : false;
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
     * @return (int|float)[]
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
                throw new InvalidArgumentException('Learner requires a'
                    . ' labeled training set.');
            }
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner init '
                . Params::stringify($this->params()));
        }

        $this->backend->flush();

        foreach ($this->experts as $estimator) {
            $this->backend->enqueue(
                new Deferred(
                    [self::class, '_train'],
                    [$estimator, $dataset]
                ),
                [$this, 'afterTrain']
            );
        }

        $this->experts = $this->backend->process();

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
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $this->backend->flush();

        foreach ($this->experts as $estimator) {
            $this->backend->enqueue(new Deferred(
                [self::class, '_predict'],
                [$estimator, $dataset]
            ));
        }

        $aggregate = array_transpose($this->backend->process());

        switch ($this->type) {
            case self::CLASSIFIER:
                return array_map([$this, 'decideClass'], $aggregate);

            case self::REGRESSOR:
                return array_map([$this, 'decideValue'], $aggregate);

            case self::ANOMALY_DETECTOR:
                return array_map([$this, 'decideAnomaly'], $aggregate);

            default:
                throw new RuntimeException('Invalid estimator type.');
        }
    }

    /**
     * Decide on a class outcome.
     *
     * @param string[] $votes
     * @return string
     */
    public function decideClass(array $votes)
    {
        $scores = array_fill_keys($this->classes, 0.0);

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
     * @param string[] $votes
     * @return string
     */
    public function decideAnomaly(array $votes) : string
    {
        $scores = array_fill(0, 2, 0.0);

        foreach ($votes as $i => $vote) {
            $scores[$vote] += $this->influences[$i];
        }

        return argmax($scores);
    }

    /**
     * Train a learner with a dataset and return it.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Learner
     */
    public static function _train(Learner $estimator, Dataset $dataset) : Learner
    {
        $estimator->train($dataset);

        return $estimator;
    }

    /**
     * The callback that executes after the training task.
     *
     * @param \Rubix\ML\Learner $estimator
     * @throws \RuntimeException
     */
    public function afterTrain(Learner $estimator) : void
    {
        if (!$estimator->trained()) {
            throw new RuntimeException('There was a problem'
                . ' training ' . Params::shortName($estimator) . '.');
        }

        if ($this->logger) {
            $this->logger->info(Params::shortName($estimator)
                . ' finished training');
        }
    }
    
    /**
     * Return the predictions from an estimator.
     *
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return mixed[]
     */
    public static function _predict(Estimator $estimator, Dataset $dataset) : array
    {
        return $estimator->predict($dataset);
    }
}
