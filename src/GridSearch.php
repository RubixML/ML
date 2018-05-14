<?php

namespace Rubix\Engine;

use Rubix\Engine\Metrics\Error;
use Rubix\Engine\Metrics\Metric;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Estimator;
use Rubix\Engine\Metrics\Classification;
use Rubix\Engine\Estimators\Predictions\Prediction;
use ReflectionClass;

class GridSearch implements Estimator
{
    /**
     * The reflector instance of the base classifier.
     *
     * @param \ReflectionClass
     */
    protected $reflector;

    /**
     * The grid of hyperparameters i.e. constructor arguments of the base classifier.
     *
     * @var array
     */
    protected $params = [
        //
    ];

    /**
     * The performance metric used to score each estimator.
     *
     * @var \Rubix\Engine\Metrics\Metric
     */
    protected $metric;

    /**
     * An array of data containing an array of parameters that were used to train
     * the model, and the score it received from the supplied performance metric.
     *
     * @var array
     */
    protected $trials = [
        //
    ];

    /**
     * The best estimator according to the performance metric.
     *
     * @var \Rubix\Engine\Estimators\Estimator
     */
    protected $estimator;

    /**
     * @param  string  $base
     * @param  array  $params
     * @param  \Rubix\Engine\Metrics\Metric  $metric
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $params, Metric $metric)
    {
        $this->reflector = new ReflectionClass($base);

        if (!in_array(Estimator::class, $this->reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base class must be an estimator.');
        }

        foreach ($params as &$options) {
            $options = (array) $options;
        }

        $this->params = $params;
        $this->metric = $metric;
    }

    /**
     * Return the underlying estimator instance.
     *
     * @return \Rubix\Engine\Estimators\Estimator
     */
    public function estimator() : Estimator
    {
        return $this->estimator;
    }

    /**
     * @return array
     */
    public function trials() : array
    {
        return $this->trials;
    }

    /**
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \RuntimeException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        $this->trials = [];

        if ($this->metric->minimize()) {
            $best = ['score' => INF, 'estimator' => null];
        } else {
            $best = ['score' => -INF, 'estimator' => null];
        }

        $dataset->randomize();

        if (!in_array(Classifier::class, $this->reflector->getInterfaceNames())) {
            list($training, $testing) = $dataset->stratifiedSplit(0.8);
        } else {
            list($training, $testing) = $dataset->split(0.8);
        }

        foreach ($this->combineParams($this->params) as $params) {
            $estimator = $this->reflector->newInstanceArgs($params);

            $estimator->train($training);

            $predictions = array_map(function ($sample) use ($estimator) {
                return $estimator->predict($sample)->outcome();
            }, $testing->samples());

            $score = $this->metric->score($predictions, $testing->outcomes());

            if ($this->metric->minimize()) {
                if ($score < $best['score']) {
                    $best = ['score' => $score, 'estimator' => $estimator];
                }
            } else {
                if ($score > $best['score']) {
                    $best = ['score' => $score, 'estimator' => $estimator];
                }
            }

            $this->trials[] = ['score' => $score, 'params' => $params];
        }

        $this->estimator = $best['estimator'];
    }

    /**
     * Call the best estimator to make a prediction.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Estimaotors\Predictions\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        return $this->estimator->predict($sample);
    }

    /**
     * Return an array of all possible combinations of parameters. i.e. the
     * Cartesian product of the supplied parameter grid.
     *
     * @param  array  $params
     * @return array
     */
    protected function combineParams(array $grid) : array
    {
        $params = [[]];

        foreach ($grid as $i => $options) {
            $append = [];

            foreach ($params as $product) {
                foreach ($options as $option) {
                    $product[$i] = $option;
                    $append[] = $product;
                }
            }

            $params = $append;
        }

        return $params;
    }

    /**
     * Allow methods to be called from the base estimator from this estimator.
     *
     * @param  string  $name
     * @param  array  $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->estimator->$name(...$arguments);
    }
}
