<?php

namespace Rubix\Engine\ModelSelection;

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Regression;
use Rubix\Engine\Metrics\Validation\Classification;
use Rubix\Engine\Estimators\Predictions\Prediction;
use ReflectionClass;

class GridSearch
{
    /**
     * The reflector instance of the base classifier.
     *
     * @param \ReflectionClass
     */
    protected $reflector;

    /**
     * The grid of hyperparameters i.e. constructor arguments of the base
     * estimator.
     *
     * @var array
     */
    protected $grid = [
        //
    ];

    /**
     * The performance metric used to score each estimator.
     *
     * @var \Rubix\Engine\Metrics\Validation\Validation
     */
    protected $metric;

    /**
     * @param  string  $base
     * @param  array  $grid
     * @param  \Rubix\Engine\Metrics\Validation\Validation  $metric
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $grid, Validation $metric)
    {
        $reflector = new ReflectionClass($base);

        if (!in_array(Estimator::class, $reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base class must be an estimator.');
        }

        foreach ($params as &$options) {
            $options = (array) $options;
        }

        $this->reflector = $reflector;
        $this->grid = $grid;
        $this->metric = $metric;
    }

    /**
     * Train one estimator per combination of parameters given by the grid and
     * return the best one. Model is evaluated using the specified metric over
     * the supplied testing set.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $training
     * @param  \Rubix\Engine\Datasets\Supervised  $testing
     * @return array
     */
    public function search(Supervised $training, Supervised $testing) : array
    {
        $best = ['score' => -INF, 'estimator' => null];

        $trials = [];

        foreach ($this->combineParams($this->grid) as $params) {
            $estimator = $this->reflector->newInstanceArgs($params);

            $estimator->train($training);

            $predictions = $estimator->predict($testing);

            $score = $this->metric->score($predictions, $testing->labels());

            if ($score > $best['score']) {
                $best = [
                    'score' => $score,
                    'estimator' => $estimator,
                ];
            }

            $trials[] = ['score' => $score, 'params' => $params];
        }

        return [
            'estimator' => $best['estimator'],
            'results' => $trials,
        ];
    }

    /**
     * Return an array of all possible combinations of parameters. i.e. the
     * Cartesian product of the supplied parameter grid.
     *
     * @param  array  $grid
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
}
