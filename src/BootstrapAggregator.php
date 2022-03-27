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

use function in_array;
use function array_count_values;

/**
 * Bootstrap Aggregator
 *
 * Bootstrap Aggregating (or *bagging* for short) is a model averaging technique designed
 * to improve the stability and performance of a user-specified base estimator by training
 * a number of them on a unique *bootstrapped* training set sampled at random with
 * replacement.
 *
 * References:
 * [1] L. Breiman. (1996). Bagging Predictors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BootstrapAggregator implements Estimator, Learner, Parallel, Persistable
{
    use AutotrackRevisions, Multiprocessing;

    /**
     * The estimator type codes that the ensemble is compatible with.
     *
     * @var list<int>
     */
    protected const COMPATIBLE_ESTIMATOR_TYPES = [
        EstimatorType::CLASSIFIER,
        EstimatorType::REGRESSOR,
        EstimatorType::ANOMALY_DETECTOR,
    ];

    /**
     * The minimum size of each training subset.
     *
     * @var int
     */
    protected const MIN_SUBSAMPLE = 1;

    /**
     * The base learner.
     *
     * @var \Rubix\ML\Learner
     */
    protected \Rubix\ML\Learner $base;

    /**
     * The number of base learners to train in the ensemble.
     *
     * @var int
     */
    protected int $estimators;

    /**
     * The ratio of samples from the training set to randomly subsample to train each base learner.
     *
     * @var float
     */
    protected float $ratio;

    /**
     * The ensemble of estimators.
     *
     * @var list<\Rubix\ML\Learner>
     */
    protected array $ensemble = [
        //
    ];

    /**
     * @param \Rubix\ML\Learner $base
     * @param int $estimators
     * @param float $ratio
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(Learner $base, int $estimators = 10, float $ratio = 0.5)
    {
        if (!in_array($base->type()->code(), self::COMPATIBLE_ESTIMATOR_TYPES)) {
            throw new InvalidArgumentException('This meta estimator'
                . ' only supports classifiers, regressors, and'
                . " anomaly detectors, {$base->type()} given.");
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Number of estimators'
                . " must be greater than 0, $estimators given.");
        }

        if ($ratio <= 0.0 or $ratio > 1.5) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0 and 1.5, $ratio given.");
        }

        $this->base = $base;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
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
        return $this->base->type();
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
        return $this->base->compatibility();
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
            'base' => $this->base,
            'estimators' => $this->estimators,
            'ratio' => $this->ratio,
        ];
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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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

        $p = max(self::MIN_SUBSAMPLE, (int) round($this->ratio * $dataset->numSamples()));

        $this->backend->flush();

        for ($i = 0; $i < $this->estimators; ++$i) {
            $estimator = clone $this->base;

            $subset = $dataset->randomSubsetWithReplacement($p);

            $task = new TrainLearner($estimator, $subset);

            $this->backend->enqueue($task);
        }

        $this->ensemble = $this->backend->process();
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->ensemble)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $this->backend->flush();

        foreach ($this->ensemble as $estimator) {
            $task = new Predict($estimator, $dataset);

            $this->backend->enqueue($task);
        }

        $aggregate = array_transpose($this->backend->process());

        switch ($this->type()) {
            case EstimatorType::classifier():
            case EstimatorType::anomalyDetector():
                return array_map([$this, 'decideDiscrete'], $aggregate);

            default:
                return array_map([Stats::class, 'mean'], $aggregate);
        }
    }

    /**
     * Decide on a discrete-valued outcome.
     *
     * @param string[] $votes
     * @return string
     */
    protected function decideDiscrete(array $votes) : string
    {
        /** @var array<string,int> $counts */
        $counts = array_count_values($votes);

        return argmax($counts);
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
        return 'Bootstrap Aggregator (' . Params::stringify($this->params()) . ')';
    }
}
