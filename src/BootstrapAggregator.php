<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use Rubix\ML\Other\Functions\Argmax;
use InvalidArgumentException;
use RuntimeException;

/**
 * Bootstrap Aggregator
 *
 * Bootstrap Aggregating (or bagging) is a model averaging technique designed to
 * improve the stability and performance of a user-specified base Estimator by
 * training a number of them on a unique bootstrapped training set. Bootstrap
 * Aggregator then collects all of their predictions and makes a final
 * prediction based on the results.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BootstrapAggregator implements MetaEstimator, Ensemble, Persistable
{
    /**
     * The class name of the base estimator.
     *
     * @var string
     */
    protected $base;

    /**
     * The type of the base estimator.
     *
     * @var int
     */
    protected $type;

    /**
     * The constructor arguments of the base estimator.
     *
     * @var array
     */
    protected $params = [
        //
    ];

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
     * @var array
     */
    protected $ensemble = [
        //
    ];

    /**
     * @param  string  $base
     * @param  array  $params
     * @param  int  $estimators
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $params, int $estimators = 10, float $ratio = 0.5)
    {
        $proxy = new $base(...$params);

        if (!$proxy instanceof Estimator) {
            throw new InvalidArgumentException('Base class must be an'
                . ' estimator.');
        }

        if ($proxy instanceof MetaEstimator) {
            throw new InvalidArgumentException('Base class cannot be a meta'
                . ' estimator.');
        }

        if ($proxy->type() !== self::CLASSIFIER and $proxy->type() !== self::REGRESSOR
            and $proxy->type() !== self::DETECTOR) {
                throw new InvalidArgumentException('This ensemble only supports'
                    . ' classifiers, regressors, and detectors.');
            }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must have at least'
                . ' 1 estimator.');
        }

        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Sample ratio must be between'
                . ' 0.01 and 1.');
        }

        $this->base = $base;
        $this->type = $proxy->type();
        $this->params = $params;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->type;
    }

    /**
     * Return the ensemble of estimators.
     *
     * @return array
     */
    public function estimators() : array
    {
        return $this->ensemble;
    }

    /**
     * Instantiate and train each base estimator in the ensemble on a bootstrap
     * training set.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $n = (int) round($this->ratio * $dataset->numRows());

        $this->ensemble = [];

        for ($epoch = 0; $epoch < $this->estimators; $epoch++) {
            $estimator = new $this->base(...$this->params);

            $subset = $dataset->randomSubsetWithReplacement($n);

            $estimator->train($subset);

            $this->ensemble[] = $estimator;
        }
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
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

        $predictions = [];

        foreach ($aggregate as $outcomes) {
            if ($this->type === self::CLASSIFIER) {
                $predictions[] = Argmax::compute(array_count_values($outcomes));
            } else if ($this->type === self::DETECTOR) {
                $predictions[] = Average::mean($outcomes) > 0.5 ? 1 : 0;
            } else {
                $predictions[] = Average::mean($outcomes);
            }
        }

        return $predictions;
    }
}
