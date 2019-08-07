<?php

namespace Rubix\ML;

use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\CrossValidation\Validator;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Other\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use ReflectionClass;

/**
 * Grid Search
 *
 * Grid Search is an algorithm that optimizes hyper-parameter selection. From
 * the user's perspective, the process of training and predicting is the same,
 * however, under the hood, Grid Search trains one estimator per combination
 * of parameters and the best model is selected as the base estimator.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GridSearch implements Estimator, Learner, Parallel, Persistable, Verbose
{
    use Multiprocessing, PredictsSingle, LoggerAware;

    /**
     * The class name of the base estimator.
     *
     * @var string
     */
    protected $base;

    /**
     * The combinations of hyperparameters i.e. constructor arguments to be
     * used to instantiate and train a base learner.
     *
     * @var array
     */
    protected $combinations = [
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
     * A tuple containing the parameters with the highest validation score and
     * the validation score.
     *
     * @var (float|array)[]|null
     */
    protected $best;

    /**
     * The instance of the estimator with the best parameters.
     *
     * @var \Rubix\ML\Learner
     */
    protected $estimator;

    /**
     * Return an array of all possible combinations of parameters. i.e the
     * Cartesian product of the supplied parameter grid.
     *
     * @param array $grid
     * @return array
     */
    public static function combineGrid(array $grid) : array
    {
        $combinations = [[]];

        foreach ($grid as $i => $params) {
            $append = [];

            foreach ($combinations as $product) {
                foreach ($params as $param) {
                    $product[$i] = $param;
                    $append[] = $product;
                }
            }

            $combinations = $append;
        }

        return $combinations;
    }

    /**
     * @param string $base
     * @param array $grid
     * @param \Rubix\ML\CrossValidation\Metrics\Metric|null $metric
     * @param \Rubix\ML\CrossValidation\Validator|null $validator
     * @throws \InvalidArgumentException
     */
    public function __construct(
        string $base,
        array $grid,
        ?Metric $metric = null,
        ?Validator $validator = null
    ) {
        $reflector = new ReflectionClass($base);

        $proxy = $reflector->newInstanceWithoutConstructor();

        if (!$proxy instanceof Learner) {
            throw new InvalidArgumentException('Base class must be an instance'
                . ' of a learner.');
        }

        $args = Params::args($proxy);

        if (count($grid) > count($args)) {
            throw new InvalidArgumentException('Too many arguments supplied'
                . ' for learner, ' . count($grid) . ' given but only '
                . count($args) . ' required.');
        }

        $grid = array_values($grid);

        foreach ($grid as $position => &$options) {
            if (!is_array($options)) {
                $options = [$options];

                continue 1;
            }

            $options = array_values($options);

            if (is_string($options[0]) or is_numeric($options[0])) {
                $options = array_unique($options);
            }
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::check($proxy, $metric);
        } else {
            switch ($proxy->type()) {
                case self::CLASSIFIER:
                    $metric = new FBeta();
                    break 1;
    
                case self::REGRESSOR:
                    $metric = new RSquared();
                    break 1;
                
                case self::CLUSTERER:
                    $metric = new VMeasure();
                    break 1;
    
                case self::ANOMALY_DETECTOR:
                    $metric = new FBeta();
                    break 1;
    
                default:
                    $metric = new Accuracy();
            }
        }

        $combinations = self::combineGrid($grid);

        $this->base = $base;
        $this->combinations = $combinations;
        $this->args = array_slice($args, 0, count($grid));
        $this->metric = $metric;
        $this->validator = $validator ?? new KFold(5);
        $this->estimator = $proxy;
        $this->backend = new Serial();
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->estimator->type();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->estimator->compatibility();
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->estimator->trained();
    }

    /**
     * The combinations of parameters from the grid.
     *
     * @return array
     */
    public function combinations() : array
    {
        return $this->combinations;
    }

    /**
     * Return the parameters that had the highest validation score.
     *
     * @return (float|array)[]|null
     */
    public function best() : ?array
    {
        return $this->best;
    }

    /**
     * Return the base estimator instance.
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner init ' . Params::stringify([
                'base' => $this->estimator,
                'metric' => $this->metric,
                'validator' => $this->validator,
                'backend' => $this->backend,
            ]));
        }

        $this->backend->flush();

        foreach ($this->combinations as $params) {
            $estimator = new $this->base(...$params);

            $this->backend->enqueue(
                new Deferred(
                    [self::class, 'score'],
                    [
                        $this->validator,
                        $estimator,
                        $dataset,
                        $this->metric,
                    ]
                ),
                function ($result) use ($params) {
                    if ($this->logger) {
                        $constructor = array_combine($this->args, $params);

                        $this->logger->info(Params::stringify([
                            'Params' => $constructor ?: [],
                            Params::shortName($this->metric) => $result,
                        ]));
                    }
                }
            );
        }

        $scores = $this->backend->process();

        $bestScore = -INF;
        $bestParams = [];

        foreach ($scores as $i => $score) {
            if ($score > $bestScore) {
                $bestParams = $this->combinations[$i];
                $bestScore = $score;
            }
        }

        $this->best = ['score' => $bestScore, 'params' => $bestParams];

        if ($this->logger) {
            $constructor = array_combine($this->args, $bestParams) ?: [];

            $this->logger->info('Best: ' . Params::stringify($constructor));
        }

        $estimator = new $this->base(...$bestParams);

        if ($this->logger) {
            $this->logger->info('Training base on full dataset');
        }

        $estimator->train($dataset);

        $this->estimator = $estimator;

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->estimator->predict($dataset);
    }

    /**
     * Cross validate a learner with a given dataset and return the score.
     *
     * @param \Rubix\ML\CrossValidation\Validator $validator
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @return float
     */
    public static function score(Validator $validator, Learner $estimator, Labeled $dataset, Metric $metric) : float
    {
        return $validator->test($estimator, $dataset, $metric);
    }

    /**
     * Allow methods to be called on the estimator from the wrapper.
     *
     * @param string $name
     * @param array $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->estimator->$name(...$arguments);
    }
}
