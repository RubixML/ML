<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Validator;
use Rubix\ML\CrossValidation\Metrics\Metric;
use InvalidArgumentException;
use RuntimeException;
use ReflectionMethod;
use ReflectionClass;

/**
 * Grid Search
 *
 * Grid Search is an algorithm that optimizes hyperparameter selection. From the
 * user’s perspective, the process of training and predicting is the same,
 * however, under the hood, Grid Search trains one Estimator per combination of
 * parameters and predictions are made using the best Estimator. You can access
 * the scores for each parameter combination by calling the results() method on
 * the trained Grid Search meta-Estimator or you can get the best parameters by
 * calling best().
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
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
     * @var \Rubix\ML\CrossValidation\Metrics\Metric
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
     * The type of estimator this meta estimator wraps.
     *
     * @var int
     */
    protected $type;

    /**
     * The results of a grid search.
     *
     * @var array
     */
    protected $results = [
        //
    ];

    /**
     * The parameters with the highest validation score.
     *
     * @var array|null
     */
    protected $best;

    /**
     * The highest validation score.
     *
     * @var float|null
     */
    protected $score;

    /**
     * The instance of the estimator with the best parameters.
     *
     * @var \Rubix\ML\Estimator
     */
    protected $estimator;

    /**
     * @param  string  $base
     * @param  array  $params
     * @param  \Rubix\ML\CrossValidation\Metrics\Metric  $metric
     * @param  \Rubix\ML\CrossValidation\Validator|null  $validator
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $params, Metric $metric, Validator $validator = null)
    {
        $reflector = new ReflectionClass($base);

        $proxy = $reflector->newInstanceWithoutConstructor();

        if (!$proxy instanceof Estimator) {
            throw new InvalidArgumentException('Base class must be an'
                . ' estimator.');
        }

        if ($proxy instanceof MetaEstimator) {
            throw new InvalidArgumentException('Base estimator cannot be a meta'
                . ' estimator.');
        }

        $constructor = $reflector->getConstructor();

        if ($constructor instanceof ReflectionMethod) {
            $args = array_column($constructor->getParameters(), 'name');
        } else {
            $args = [];
        }

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
        $this->estimator = $proxy;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->estimator->type();
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
     * @return array|null
     */
    public function best() : ?array
    {
        return $this->best;
    }

    /**
     * Return the highest validation score.
     *
     * @return float|null
     */
    public function score() : ?float
    {
        return $this->score;
    }

    /**
     * Return the best estimator instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function estimator() : Estimator
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

        $bestScore = -INF;
        $bestParams = [];
        $bestEstimator = null;

        foreach ($this->combineParams($this->params) as $params) {
            $estimator = new $this->base(...$params);

            $score = $this->validator->test($estimator, $dataset, $this->metric);

            if ($score > $bestScore) {
                $bestScore = $score;
                $bestParams = $params;
                $bestEstimator = $estimator;
            }

            $this->results[] = [
                'score' => $score,
                'params' => array_combine($this->args, $params),
            ];
        }

        $this->best = $bestParams;
        $this->score = $bestScore;
        $this->estimator = $bestEstimator;
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
