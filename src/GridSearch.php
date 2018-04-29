<?php

namespace Rubix\Engine;

use Rubix\Engine\Metrics\Error;
use Rubix\Engine\Metrics\Metric;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Metrics\Classification;
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
     * The ratio of samples to leave out for validation. A higher ratio will
     * result in less data being used to train the model, however the test
     * results will have lower variance.
     *
     * @var float
     */
    protected $ratio;

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
     * @var \Rubix\Engine\Estimator
     */
    protected $estimator;

    /**
     * @param  string  $base
     * @param  array  $params
     * @param  \Rubix\Engine\Metrics\Metric  $metric
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $params, Metric $metric, float $ratio = 0.2)
    {
        if ($ratio < 0.01 || $ratio > 1) {
            throw new InvalidArgumentException('Testing ratio must be a float value between 0.01 and 1.0.');
        }

        $this->reflector = new ReflectionClass($base);

        if (!in_array(Estimator::class, $this->reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base class must be an estimator.');
        }

        foreach ($params as &$options) {
            $options = (array) $options;
        }

        $this->params = $params;
        $this->metric = $metric;
        $this->ratio = $ratio;
    }

    /**
     * Return the underlying estimator instance.
     *
     * @return \Rubix\Engine\Estimator
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
        } else if ($this->metric instanceof Error) {
            $best = ['score' => 0, 'estimator' => null];
        }

        foreach ($this->combineParams($this->params) as $params) {
            list($training, $testing) = $dataset->split($this->ratio);

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
     * @return \Rubix\Engine\Prediction
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
