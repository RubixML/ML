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
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use ReflectionClass;

use function count;
use function array_slice;

/**
 * Grid Search
 *
 * Grid Search is an algorithm that optimizes hyper-parameter selection. From
 * the user's perspective, the process of training and predicting is the same,
 * however, under the hood, Grid Search trains one estimator per combination
 * of parameters and the best model is selected as the base estimator.
 *
 * > **Note:** You can choose the hyper-parameters manually or you can generate
 * them randomly or in a grid using the Params helper.
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
     * An array of tuples where each tuple contains the possible values of each
     * parameter in the order they are given to the base learner's constructor.
     *
     * @var array[]
     */
    protected $grid;

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
     * @var string[]
     */
    protected $args = [
        //
    ];

    /**
     * A 2-tuple containing the parameters with the highest validation score
     * and the validation score.
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
     * @param class-string $base
     * @param array[] $grid
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

        foreach ($grid as &$options) {
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

        $this->base = $base;
        $this->grid = $grid;
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
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        if (!$this->trained()) {
            return DataType::all();
        }

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
            throw new InvalidArgumentException('Learner requires a'
                . ' labeled training set.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner init ' . Params::stringify([
                'base' => $this->estimator,
                'metric' => $this->metric,
                'validator' => $this->validator,
                'backend' => $this->backend,
            ]));
        }

        $this->backend->flush();

        $combinations = $this->combinations();

        foreach ($combinations as $params) {
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
                $bestParams = $combinations[$i];
                $bestScore = $score;
            }
        }

        $this->best = $bestParams;

        $estimator = new $this->base(...$bestParams);

        if ($this->logger) {
            $this->logger->info('Training learner with best'
                . ' params on full dataset');
        }

        $estimator->train($dataset);

        $this->estimator = $estimator;

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Return an array of all possible combinations of parameters. i.e the
     * Cartesian product of the user supplied parameter grid.
     *
     * @return array[]
     */
    public function combinations() : array
    {
        $combinations = [[]];

        foreach ($this->grid as $i => $params) {
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
     * Make a prediction on a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return mixed[]
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
     * @param mixed[] $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->estimator->$name(...$arguments);
    }
}
