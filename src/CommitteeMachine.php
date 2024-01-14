<?php

namespace Rubix\ML;

use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\Multiprocessing;
use Rubix\ML\Backends\Tasks\Predict;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Backends\Tasks\TrainLearner;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function in_array;

/**
 * Committee Machine
 *
 * A voting ensemble that aggregates the predictions of a committee of heterogeneous
 * estimators (called *experts*). The committee uses a user-specified influence-based
 * scheme to weight the final predictions.
 *
 * > **Note**: Influence values can be arbitrary as they are normalized upon instantiation.
 *
 * References:
 * [1] H. Drucker. (1997). Fast Committee Machines for Regression and Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CommitteeMachine implements Estimator, Learner, Parallel, Persistable
{
    use AutotrackRevisions, Multiprocessing;

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
    protected array $experts;

    /**
     * The influence values of each expert in the committee.
     *
     * @var list<float>
     */
    protected array $influences;

    /**
     * The data types that the committee is compatible with.
     *
     * @var list<\Rubix\ML\DataType>
     */
    protected array $compatibility;

    /**
     * The zero vector of each possible discrete outcome.
     *
     * @var float[]
     */
    protected array $classes = [
        //
    ];

    /**
     * @param \Rubix\ML\Learner[] $experts
     * @param (int|float)[]|null $influences
     * @throws InvalidArgumentException
     */
    public function __construct(array $experts, ?array $influences = null)
    {
        if (empty($experts)) {
            throw new InvalidArgumentException('Committee must contain at least 1 expert.');
        }

        $prototype = current($experts);

        $compatibilities = [];

        foreach ($experts as $expert) {
            if (!$expert instanceof Learner) {
                throw new InvalidArgumentException('Expert must implement the Learner interface.');
            }

            if (!in_array($expert->type()->code(), self::COMPATIBLE_ESTIMATOR_TYPES)) {
                throw new InvalidArgumentException('Committee only supports'
                    . ' classifiers, regressors, and anomaly detectors, '
                    . " {$expert->type()} given.");
            }

            if ($expert->type() != $prototype->type()) {
                throw new InvalidArgumentException('Experts must be of'
                    . " the same type, {$prototype->type()} expected but"
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
                throw new InvalidArgumentException('Number of influences'
                    . " must be equal to the number of experts, $k"
                    . ' expected but ' . count($influences) . ' given.');
            }

            $total = array_sum($influences);

            if ($total <= 0) {
                throw new InvalidArgumentException('Total influence must'
                    . " be greater than 0, $total given.");
            }

            foreach ($influences as &$influence) {
                $influence /= $total;
            }

            $influences = array_values($influences);
        } else {
            $influences = array_fill(0, $k, 1.0 / $k);
        }

        $this->experts = array_values($experts);
        $this->influences = $influences;
        $this->compatibility = $compatibility;
        $this->backend = new Serial();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
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
     * @return list<\Rubix\ML\Learner>
     */
    public function experts() : array
    {
        return $this->experts;
    }

    /**
     * Return the normalized influences for each expert in the committee.
     *
     * @return list<float>
     */
    public function influences() : array
    {
        return $this->influences;
    }

    /**
     * Train all the experts with the dataset.
     *
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        $specifications = [
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ];

        if ($this->type()->isSupervised()) {
            $specifications[] = new DatasetIsLabeled($dataset);

            if ($dataset instanceof Labeled) {
                $specifications[] = new LabelsAreCompatibleWithLearner($dataset, $this);
            }
        }

        SpecificationChain::with($specifications)->check();

        $this->backend->flush();

        foreach ($this->experts as $estimator) {
            $task = new TrainLearner($estimator, $dataset);

            $this->backend->enqueue($task);
        }

        $this->experts = $this->backend->process();

        switch ($this->type()) {
            case EstimatorType::classifier():
                if ($dataset instanceof Labeled) {
                    $this->classes = array_fill_keys($dataset->possibleOutcomes(), 0.0);
                }

                break;

            case EstimatorType::anomalyDetector():
                $this->classes = [0 => 0.0, 1 => 0.0];

                break;
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param Dataset $dataset
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $this->backend->flush();

        foreach ($this->experts as $estimator) {
            $task = new Predict($estimator, $dataset);

            $this->backend->enqueue($task);
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
     * @param list<int|string> $votes
     * @return string|int
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
     * @param list<int|float> $votes
     * @return float
     */
    protected function decideContinuous(array $votes) : float
    {
        return Stats::weightedMean($votes, $this->influences);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Committee Machine (' . Params::stringify($this->params()) . ')';
    }
}
