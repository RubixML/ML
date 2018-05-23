<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\CrossValidation\Validator;
use InvalidArgumentException;
use ReflectionClass;

class GridSearch implements Estimator, Persistable
{
    /**
     * The reflector instance of the base estimator.
     *
     * @var \ReflectionClass
     */
    protected $reflector;

    /**
     * The argument names for the base estimator's constructor.
     *
     * @var array
     */
    protected $args = [
        //
    ];

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
     * The validator used to score each trained estimator.
     *
     * @var \Rubix\Engine\CrossValidation\Validator
     */
    protected $validator;

    /**
     * The results of a grid search.
     *
     * @var array
     */
    protected $results = [
        //
    ];

    /**
     * @param  string  $base
     * @param  array  $grid
     * @param  \Rubix\Engine\CrossValidation\Validator  $validator
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $grid, Validator $validator)
    {
        $reflector = new ReflectionClass($base);

        if (!in_array(Estimator::class, $reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base class must implement the'
                . ' estimator inteferace.');
        }

        $args = array_column($reflector->getConstructor()
            ->getParameters(), 'name');

        if (count($grid) > count($args)) {
            throw new InvalidArgumentException('Too many arguments supplied.'
                . count($grid) . ' given, only ' . count($args) . ' needed.');
        }

        foreach ($grid as &$params) {
            $params = (array) $params;
        }

        $this->reflector = $reflector;
        $this->args = array_slice($args, 0, count($grid));
        $this->grid = $grid;
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
     * Train one estimator per combination of parameters given by the grid and
     * assign the best one as the base estimator of this instance.
     *
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @return void
     */
    public function train(Labeled $dataset) : void
    {
        $best = ['score' => -INF, 'estimator' => null];

        $this->results = [];

        foreach ($this->combineParams($this->grid) as $params) {
            $estimator = $this->reflector->newInstanceArgs($params);

            $score = $this->validator->score($estimator, $dataset);

            if ($score > $best['score']) {
                $best = ['score' => $score, 'estimator' => $estimator];
            }

            $this->results[] = [
                'score' => $score,
                'params' => array_combine($this->args, $params),
            ];
        }

        $this->estimator = $best['estimator'];
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        return $this->estimator->predict($samples);
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
