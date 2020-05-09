<?php

namespace Rubix\ML;

use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Backends\Tasks\Predict;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Backends\Tasks\TrainLearner;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function in_array;

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
    use Multiprocessing, PredictsSingle;

    /**
     * The estimator type codes that the ensemble is compatible with.
     *
     * @var int[]
     */
    protected const COMPATIBLE_ESTIMATOR_TYPES = [
        EstimatorType::CLASSIFIER,
        EstimatorType::REGRESSOR,
        EstimatorType::ANOMALY_DETECTOR,
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
     * The ensemble of estimators.
     *
     * @var \Rubix\ML\Learner[]
     */
    protected $ensemble = [
        //
    ];

    /**
     * @param \Rubix\ML\Learner $base
     * @param int $estimators
     * @param float $ratio
     * @throws \InvalidArgumentException
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
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->base->type();
    }

    /**
     * Return the data types that the model is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
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
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if ($this->type()->isClassifier() or $this->type()->isRegressor()) {
            if (!$dataset instanceof Labeled) {
                throw new InvalidArgumentException('Learner requires a'
                    . ' Labeled training set.');
            }
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $p = (int) ceil($this->ratio * $dataset->numRows());

        $this->backend->flush();

        for ($i = 0; $i < $this->estimators; ++$i) {
            $estimator = clone $this->base;

            $subset = $dataset->randomSubsetWithReplacement($p);

            $this->backend->enqueue(new TrainLearner($estimator, $subset));
        }

        $this->ensemble = $this->backend->process();
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->ensemble)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $this->backend->flush();

        foreach ($this->ensemble as $estimator) {
            $this->backend->enqueue(new Predict($estimator, $dataset));
        }

        $aggregate = array_transpose($this->backend->process());
        
        switch ($this->type()) {
            case EstimatorType::classifier():
            case EstimatorType::anomalyDetector():
                return array_map([self::class, 'decideDiscrete'], $aggregate);

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
    public function decideDiscrete(array $votes) : string
    {
        return argmax(array_count_values($votes));
    }
}
