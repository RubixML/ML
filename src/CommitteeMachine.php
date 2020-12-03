<?php

namespace Rubix\ML;

use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Helpers\Verifier;
use Rubix\ML\Backends\Tasks\Predict;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Backends\Tasks\TrainLearner;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;
use Stringable;

use function count;
use function in_array;

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
class CommitteeMachine implements Estimator, Learner, Parallel, Verbose, Persistable, Stringable
{
    use Multiprocessing, PredictsSingle, LoggerAware;

    /**
     * The integer-encoded estimator types this ensemble is compatible with.
     *
     * @var list<int>
     */
    protected const COMPATIBLE_ESTIMATOR_TYPES = [
        EstimatorType::CLASSIFIER,
        EstimatorType::REGRESSOR,
        EstimatorType::ANOMALY_DETECTOR,
    ];

    /**
     * The committee of experts. i.e. the ensemble of estimators.
     *
     * @var list<\Rubix\ML\Learner>
     */
    protected $experts;

    /**
     * The influence values of each expert in the committee.
     *
     * @var (int|float)[]
     */
    protected $influences;

    /**
     * The data types that the committee is compatible with.
     *
     * @var list<\Rubix\ML\DataType>
     */
    protected $compatibility;

    /**
     * The zero vector of each possible discrete outcome.
     *
     * @var float[]
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

        $proto = current($experts);

        $compatibilities = [];

        foreach ($experts as $expert) {
            if (!$expert instanceof Learner) {
                throw new InvalidArgumentException('Expert must implement'
                    . ' the Learner interface.');
            }

            if (!in_array($expert->type()->code(), self::COMPATIBLE_ESTIMATOR_TYPES)) {
                throw new InvalidArgumentException('Committee only supports'
                    . ' classifiers, regressors, and anomaly detectors, '
                    . " {$expert->type()} given.");
            }

            if ($expert->type() != $proto->type()) {
                throw new InvalidArgumentException('Experts must all be of'
                    . " the same type, {$proto->type()} expected but"
                    . " {$expert->type()} given.");
            }

            $compatibilities[] = $expert->compatibility();
        }

        $compatibility = array_values(array_intersect(...$compatibilities));

        if (count($compatibility) < 1) {
            throw new InvalidArgumentException('Committee must have at'
                . ' least 1 compatible data type in common.');
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
        $this->compatibility = $compatibility;
        $this->backend = new Serial();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->experts[array_key_first($this->experts)]->type();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return $this->compatibility;
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
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
        if ($this->type()->isSupervised()) {
            if (!$dataset instanceof Labeled) {
                throw new InvalidArgumentException('Learner requires a'
                    . ' Labeled training set.');
            }
        }

        Verifier::check([
            DatasetIsNotEmpty::with($dataset),
            SamplesAreCompatibleWithEstimator::with($dataset, $this),
        ]);

        if ($this->logger) {
            $this->logger->info("Learner init $this");

            $this->logger->info('Training started');
        }

        $this->backend->flush();

        foreach ($this->experts as $estimator) {
            $this->backend->enqueue(
                new TrainLearner($estimator, $dataset),
                [$this, 'afterTrain']
            );
        }

        $this->experts = $this->backend->process();

        switch ($this->type()) {
            case EstimatorType::classifier():
                if ($dataset instanceof Labeled) {
                    $this->classes = array_fill_keys($dataset->possibleOutcomes(), 0.0);
                }

                break 1;

            case EstimatorType::anomalyDetector():
                $this->classes = [0 => 0.0, 1 => 0.0];

                break 1;
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * The callback that executes after the training task.
     *
     * @internal
     *
     * @param \Rubix\ML\Learner $estimator
     * @throws \RuntimeException
     */
    public function afterTrain(Learner $estimator) : void
    {
        if (!$estimator->trained()) {
            throw new RuntimeException("There was a problem training $estimator.");
        }

        if ($this->logger) {
            $this->logger->info("$estimator finished training");
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
            $this->backend->enqueue(new Predict($estimator, $dataset));
        }

        $aggregate = array_transpose($this->backend->process());

        switch ($this->type()) {
            case EstimatorType::classifier():
            case EstimatorType::anomalyDetector():
                return array_map([$this, 'decideDiscrete'], $aggregate);

            default:
                return array_map([$this, 'decideContinuous'], $aggregate);
        }
    }

    /**
     * Decide on a discrete outcome.
     *
     * @param (int|string)[] $votes
     * @return string
     */
    protected function decideDiscrete(array $votes)
    {
        $scores = $this->classes;

        foreach ($votes as $i => $vote) {
            $scores[$vote] += $this->influences[$i];
        }

        return argmax($scores);
    }

    /**
     * Decide on a real-valued outcome.
     *
     * @param (int|float)[] $votes
     * @return float
     */
    protected function decideContinuous(array $votes) : float
    {
        return Stats::weightedMean($votes, $this->influences);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Committee Machine (' . Params::stringify($this->params()) . ')';
    }
}
