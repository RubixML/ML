<?php

namespace Rubix\ML;

use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Backends\Tasks\Task;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\CrossValidation\Validator;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;

use function count;

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
class GridSearch implements Estimator, Learner, Parallel, Persistable, Verbose, Wrapper
{
    use Multiprocessing, PredictsSingle, LoggerAware;

    /**
     * The class name of the base estimator.
     *
     * @var string
     */
    protected $base;

    /**
     * An array of tuples containing the possible values for each of the base learner's
     * constructor parameters.
     *
     * @var array[]
     */
    protected $params;

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
     * The results of the last hyper-parameter search.
     *
     * @var array[]|null
     */
    protected $results;

    /**
     * The instance of the estimator with the best parameters.
     *
     * @var \Rubix\ML\Learner
     */
    protected $estimator;

    /**
     * @param class-string $base
     * @param array[] $params
     * @param \Rubix\ML\CrossValidation\Metrics\Metric|null $metric
     * @param \Rubix\ML\CrossValidation\Validator|null $validator
     * @throws \InvalidArgumentException
     */
    public function __construct(
        string $base,
        array $params,
        ?Metric $metric = null,
        ?Validator $validator = null
    ) {
        if (!class_exists($base)) {
            throw new InvalidArgumentException("Class $base does not exist.");
        }

        $proxy = new $base(...array_map('current', $params));

        if (!$proxy instanceof Learner) {
            throw new InvalidArgumentException('Base class must'
                . ' implement the Learner Interface.');
        }

        foreach ($params as &$tuple) {
            if (empty($tuple)) {
                $tuple = [null];
            } else {
                $tuple = array_values(array_unique($tuple, SORT_REGULAR));
            }
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::check($proxy, $metric);
        } else {
            switch ($proxy->type()) {
                case EstimatorType::classifier():
                    $metric = new FBeta(1.0);

                    break 1;
    
                case EstimatorType::regressor():
                    $metric = new RMSE();

                    break 1;
                
                case EstimatorType::clusterer():
                    $metric = new VMeasure();

                    break 1;
    
                case EstimatorType::anomalyDetector():
                    $metric = new FBeta(1.0);
                    
                    break 1;
    
                default:
                    $metric = new Accuracy();
            }
        }

        $this->base = $base;
        $this->params = $params;
        $this->metric = $metric;
        $this->validator = $validator ?? new KFold(3);
        $this->estimator = $proxy;
        $this->backend = new Serial();
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->estimator->type();
    }

    /**
     * Return the data types that the model is compatible with.
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
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'base' => $this->base,
            'params' => $this->params,
            'metric' => $this->metric,
            'validator' => $this->validator,
        ];
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
     * Return an array containing the validation scores and hyper-parameters under test
     * for each combination resulting from the last search.
     *
     * @return array[]|null
     */
    public function results() : ?array
    {
        return $this->results;
    }

    /**
     * Return an array containing the best parameters from the last search.
     *
     * @return mixed[]|null
     */
    public function best() : ?array
    {
        return $this->results ? $this->results[0]['params'] : null;
    }

    /**
     * Return the base estimator instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function base() : Estimator
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
                . ' Labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $combinations = $this->combinations();

        if ($this->logger) {
            $this->logger->info('Learner init ' . Params::stringify($this->params()));

            $this->logger->info('Searching ' . count($combinations)
                . ' combinations of hyper-parameters');
        }

        $this->backend->flush();

        $this->results = [];

        foreach ($combinations as $params) {
            $estimator = new $this->base(...$params);

            $this->backend->enqueue(
                new Task(
                    [self::class, 'score'],
                    [
                        $estimator,
                        $dataset,
                        $this->validator,
                        $this->metric,
                    ]
                ),
                [$this, 'afterScore']
            );
        }

        [$scores, $params] = array_transpose($this->backend->process());

        array_multisort($scores, $combinations, SORT_DESC);

        if ($this->logger) {
            $this->logger->info('Training base learner');
        }

        $estimator = new $this->base(...reset($combinations));

        $estimator->train($dataset);

        $this->estimator = $estimator;

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Return an array of all possible combinations of parameters. i.e the
     * Cartesian product of the user-supplied parameter array.
     *
     * @return array[]
     */
    public function combinations() : array
    {
        $combinations = [[]];

        foreach ($this->params as $i => $tuple) {
            $append = [];

            foreach ($combinations as $product) {
                foreach ($tuple as $param) {
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
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\CrossValidation\Validator $validator
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @return mixed[]
     */
    public static function score(Learner $estimator, Labeled $dataset, Validator $validator, Metric $metric) : array
    {
        $score = $validator->test($estimator, $dataset, $metric);

        return [
            'score' => $score,
            'params' => $estimator->params(),
        ];
    }

    /**
     * The callback that executes after the scoring task.
     *
     * @param mixed[] $result
     */
    public function afterScore(array $result) : void
    {
        if ($this->logger) {
            $this->logger->info(Params::stringify($result));
        }

        $this->results[] = $result;
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
