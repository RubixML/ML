<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Validator;
use Rubix\ML\CrossValidation\Metrics\Validation;
use InvalidArgumentException;
use ReflectionClass;

class GridSearch implements MetaEstimator, Persistable
{
    /**
     * The class name of the base estimator.
     *
     * @var string
     */
    protected $base;

    /**
     * The grid of hyperparameters i.e. constructor arguments of the base
     * estimator.
     *
     * @var array
     */
    protected $params = [
        //
    ];

    /**
     * The validation metric used to score the estimator.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Validation
     */
     protected $metric;

    /**
     * The validator used to test the estimator.
     *
     * @var \Rubix\ML\CrossValidation\Validator
     */
    protected $validator;

    /**
     * The argument names for the base estimator's constructor.
     *
     * @var array
     */
    protected $args = [
        //
    ];

    /**
     * The results of a grid search.
     *
     * @var array
     */
    protected $results = [
        //
    ];

    /**
     * The parameters of the best estimator.
     *
     * @var array
     */
    protected $best = [
        //
    ];

    /**
     * The instance of the estimator with the best parameters.
     *
     * @var \Rubix\ML\Estimator|null
     */
    protected $estimator;

    /**
     * @param  string  $base
     * @param  array  $params
     * @param  \Rubix\ML\CrossValidation\Metrics\Validation  $metric
     * @param  \Rubix\ML\CrossValidation\Validator|null  $validator
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $params, Validation $metric, Validator $validator = null)
    {
        $reflector = new ReflectionClass($base);

        if (!in_array(Estimator::class, $reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base class must implement the'
                . ' estimator interface.');
        }

        if (in_array(MetaEstimator::class, $reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base estimator cannot be a meta'
                . ' estimator.');
        }

        $args = array_column($reflector->getConstructor()->getParameters(),
            'name');

        if (count($params) > count($args)) {
            throw new InvalidArgumentException('Too many arguments supplied.'
                . count($params) . ' given, only ' . count($args) . ' needed.');
        }

        if (is_null($validator)) {
            $validator = new KFold(10);
        }

        $this->base = $base;
        $this->args = array_slice($args, 0, count($params));
        $this->params = $params;
        $this->metric = $metric;
        $this->validator = $validator;
    }

    /**
     * The results of the last grid search.
     *
     * @return array
     */
    public function results() : array
    {
        return $this->results;
    }

    /**
     * Return the parameters that had the highest validation score.
     *
     * @return array
     */
    public function best(): array
    {
        return $this->best;
    }

    /**
     * Return the best estimator instance.
     *
     * @return \Rubix\ML\Estimator|null
     */
    public function estimator() : ?Estimator
    {
        return $this->estimator;
    }

    /**
     * Train one estimator per combination of parameters given by the grid and
     * assign the best one as the base estimator of this instance.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $this->results = $this->best = [];

        $best = ['score' => -INF, 'params' => []];

        foreach ($this->combineParams($this->params) as $params) {
            $estimator = new $this->base(...$params);

            $score = $this->validator->test($estimator, $dataset,
                $this->metric);

            if ($score > $best['score']) {
                $best['score'] = $score;
                $best['params'] = $params;
            }

            $this->results[] = [
                'score' => $score,
                'params' => array_combine($this->args, $params),
            ];
        }

        $this->best = $best['params'];
        $this->estimator = new $this->base(...$best['params']);

        $this->estimator->train($dataset);
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->estimator->predict($dataset);
    }

    /**
     * Return an array of all possible combinations of parameters. i.e. the
     * Cartesian product of the supplied parameter grid.
     *
     * @param  array  $params
     * @return array
     */
    protected function combineParams(array $params) : array
    {
        $temp = [[]];

        foreach ($params as $i => $options) {
            $append = [];

            foreach ($temp as $product) {
                foreach ($options as $option) {
                    $product[$i] = $option;
                    $append[] = $product;
                }
            }

            $temp = $append;
        }

        return $temp;
    }

    /**
     * Allow methods to be called on the estimator from the wrapper.
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
