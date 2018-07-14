<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use Rubix\ML\Regressors\Regressor;
use Rubix\ML\Clusterers\Clusterer;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\AnomalyDetectors\Detector;
use InvalidArgumentException;
use ReflectionClass;

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
     * The base estimator reflector instance.
     *
     * @var \ReflectionClass
     */
    protected $reflector;

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
        $reflector = new ReflectionClass($base);

        if (!in_array(Estimator::class, $reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base class must implement the'
                . ' estimator interface.');
        }

        if (in_array(Clusterer::class, $reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('This meta estimator does not'
                . ' work with clusterers.');
        }

        if (in_array(MetaEstimator::class, $reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base estimator cannot be a meta'
                . ' estimator.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . ' 1 estimator.');
        }

        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Sample ratio must be between'
                . ' 0.01 and 1.');
        }

        $this->base = $base;
        $this->reflector = $reflector;
        $this->params = $params;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
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
        $n = (int) ($this->ratio * $dataset->numRows());

        $this->ensemble = [];

        for ($epoch = 0; $epoch < $this->estimators; $epoch++) {
            $estimator = new $this->base(...$this->params);

            $estimator->train($dataset->randomSubsetWithReplacement($n));

            $this->ensemble[] = $estimator;
        }
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $n = $dataset->numRows();

        $predictions = [[]];

        foreach ($this->ensemble as $estimator) {
            foreach ($estimator->predict($dataset) as $i => $prediction) {
                $predictions[$i][] = $prediction;
            }
        }

        return $this->aggregate($predictions);
    }

    /**
     * Aggregate the predictions made by the ensemble into a single prediction
     * based on the type of the base estimator.
     *
     * @param  array  $predictions
     * @return array
     */
    protected function aggregate(array $predictions) : array
    {
        if (in_array(Classifier::class, $this->reflector->getInterfaceNames())) {
            return array_map(function ($outcomes) {
                $counts = array_count_values($outcomes);

                return array_search(max($counts), $counts);
            }, $predictions);
        } else if (in_array(Detector::class, $this->reflector->getInterfaceNames())) {
            return array_map(function ($outcomes) {
                return Average::mean($outcomes) >= 0.5 ? 1 : 0;
            }, $predictions);
        } else {
            return array_map(function ($outcomes) {
                return Average::median($outcomes);
            }, $predictions);
        }
    }
}
