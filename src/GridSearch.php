<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Other\Traits\LoggerAware;
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
class GridSearch implements MetaEstimator, Learner, Verbose, Persistable
{
    use LoggerAware;

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
    protected $grid = [
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
     * Should we retrain the best estimator using the whole dataset?
     * 
     * @var bool
     */
    protected $retrain;

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
     * Every combination of parmeters from the last grid search.
     *
     * @var array
     */
    protected $params = [
        //
    ];

    /**
     * The validation scores of each parmeter search.
     *
     * @var array
     */
    protected $scores = [
        //
    ];

    /**
     * A tuple containing the parameters with the highest validation score and
     * the validation score.
     *
     * @var array|null
     */
    protected $best;

    /**
     * The instance of the estimator with the best parameters.
     *
     * @var \Rubix\ML\Estimator
     */
    protected $estimator;

    /**
     * @param  string  $base
     * @param  array  $grid
     * @param  \Rubix\ML\CrossValidation\Metrics\Metric  $metric
     * @param  \Rubix\ML\CrossValidation\Validator|null  $validator
     * @param  bool  $retrain
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $grid, Metric $metric, Validator $validator = null,
                                bool $retrain = true)
    {
        $reflector = new ReflectionClass($base);

        $proxy = $reflector->newInstanceWithoutConstructor();

        if (!$proxy instanceof Learner) {
            throw new InvalidArgumentException('Base class must be an instance'
                . ' of a learner.');
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

        if (count($grid) > count($args)) {
            throw new InvalidArgumentException('Too many arguments supplied.'
                . count($grid) . ' given, only ' . count($args) . ' needed.');
        }

        foreach ($grid as &$options) {
            if (!is_array($options)) {
                $options = (array) $options;
            }
        }

        if (is_null($validator)) {
            $validator = new KFold();
        }

        $this->base = $base;
        $this->grid = $grid;
        $this->args = array_slice($args, 0, count($grid));
        $this->metric = $metric;
        $this->validator = $validator;
        $this->retrain = $retrain;
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
     * The combination of parameters from the last grid search.
     *
     * @return array
     */
    public function params() : array
    {
        return $this->params;
    }

    /**
     * The validation scores from the last grid search.
     *
     * @return array
     */
    public function scores() : array
    {
        return $this->scores;
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

        !isset($this->logger) ?: $this->logger->info('Search started');

        $this->params = $this->scores = $this->best = [];

        $bestScore = -INF;
        $bestParams = [];
        $bestEstimator = null;

        foreach ($this->combineGrid($this->grid) as $params) {
            $estimator = new $this->base(...$params);

            $args = $this->extractArgs($params);

            if (isset($this->logger)) {
                $str = '';

                foreach ($args as $arg => $param) {
                    $str .= $arg . '=' . (string) $param . ' ';
                }

                $this->logger->info('Testing parameters: ' . trim($str));
            }

            $score = $this->validator->test($estimator, $dataset, $this->metric);

            if ($score > $bestScore) {
                $bestScore = $score;
                $bestParams = $params;
                $bestEstimator = $estimator;
            }

            $this->params[] = $args;
            $this->scores[] = $score;

            !isset($this->logger) ?: $this->logger->info("Test score: $score");
        }

        $this->best = [
            'score' => $bestScore,
            'params' => array_combine($this->args, $bestParams),
        ];

        if ($this->retrain === true) {
            !isset($this->logger) ?: $this->logger->info("Retraining best"
                . " estimator on the full dataset");

            $bestEstimator->train($dataset);
        }

        $this->estimator = $bestEstimator;

        !isset($this->logger) ?: $this->logger->info("Search complete");
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
     * Return an array of all possible combinations of parameters. i.e the
     * Cartesian product of the supplied parameter grid.
     *
     * @param  array  $params
     * @return array
     */
    protected function combineGrid(array $params) : array
    {
        $combinations = [[]];

        foreach ($params as $i => $options) {
            $append = [];

            foreach ($combinations as $product) {
                foreach ($options as $option) {
                    $product[$i] = $option;
                    $append[] = $product;
                }
            }

            $combinations = $append;
        }

        return $combinations;
    }

    /**
     * Extract the arguments from the model constructor for display.
     * 
     * @param  array  $params
     * @return array
     */
    public function extractArgs(array $params) : array
    {
        foreach ($params as &$param) {
            if (is_object($param)) {
                $param = get_class($param);
            }

            if (is_array($param)) {
                foreach ($param as &$subParam) {
                    if (is_object($subParam)) {
                        $subParam = get_class($subParam);
                    }
                }
            }
        }

        return array_combine($this->args, $params) ?: [];
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
