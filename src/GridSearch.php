<?php

namespace Rubix\ML;

use Rubix\ML\Helpers\Params;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Traits\Multiprocessing;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\CrossValidation\Validator;
use Rubix\ML\Backends\Tasks\CrossValidate;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;

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
class GridSearch implements Estimator, Learner, Parallel, Verbose, Persistable
{
    use AutotrackRevisions, Multiprocessing, LoggerAware;

    /**
     * The class name of the base estimator.
     *
     * @var string
     */
    protected string $class;

    /**
     * An array of lists containing the possible values for each of the base learner's constructor parameters.
     *
     * @var list<list<mixed>>
     */
    protected array $params;

    /**
     * The validation metric used to score the estimator.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Metric
     */
    protected \Rubix\ML\CrossValidation\Metrics\Metric $metric;

    /**
     * The validator used to test the estimator.
     *
     * @var \Rubix\ML\CrossValidation\Validator
     */
    protected \Rubix\ML\CrossValidation\Validator $validator;

    /**
     * The base estimator instance.
     *
     * @var \Rubix\ML\Learner
     */
    protected \Rubix\ML\Learner $base;

    /**
     * The validation scores obtained from the last search.
     *
     * @var list<float>|null
     */
    protected ?array $scores = null;

    /**
     * Return an array of all possible combinations of parameters. i.e their Cartesian product.
     *
     * @param list<list<mixed>> $params
     * @return list<list<mixed>>
     */
    protected static function combine(array $params) : array
    {
        $combinations = [[]];

        /** @var int<0,max> $i */
        foreach ($params as $i => $params) {
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
     * @param class-string $class
     * @param array<mixed[]> $params
     * @param \Rubix\ML\CrossValidation\Metrics\Metric|null $metric
     * @param \Rubix\ML\CrossValidation\Validator|null $validator
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        string $class,
        array $params,
        ?Metric $metric = null,
        ?Validator $validator = null
    ) {
        if (!class_exists($class)) {
            throw new InvalidArgumentException("Class $class does not exist.");
        }

        $proxy = new $class(...array_map('current', $params));

        if (!$proxy instanceof Learner) {
            throw new InvalidArgumentException('Base class must'
                . ' implement the Learner Interface.');
        }

        $params = array_values($params);

        foreach ($params as &$tuple) {
            $tuple = empty($tuple) ? [null] : array_unique($tuple, SORT_REGULAR);
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::with($proxy, $metric)->check();
        } else {
            switch ($proxy->type()) {
                case EstimatorType::classifier():
                    $metric = new FBeta();

                    break;

                case EstimatorType::regressor():
                    $metric = new RMSE();

                    break;

                case EstimatorType::clusterer():
                    $metric = new VMeasure();

                    break;

                case EstimatorType::anomalyDetector():
                    $metric = new FBeta();

                    break;

                default:
                    $metric = new Accuracy();
            }
        }

        $this->class = $class;
        $this->params = $params;
        $this->metric = $metric;
        $this->validator = $validator ?? new KFold(3);
        $this->base = $proxy;
        $this->backend = new Serial();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->base->type();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return $this->trained()
            ? $this->base->compatibility()
            : DataType::all();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'class' => $this->class,
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
        return $this->base->trained();
    }

    /**
     * Return the base learner instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function base() : Estimator
    {
        return $this->base;
    }

    /**
     * Train one estimator per combination of parameters given by the grid and
     * assign the best one as the base estimator of this instance.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        $combinations = self::combine($this->params);

        $this->backend->flush();

        foreach ($combinations as $params) {
            /** @var \Rubix\ML\Learner $estimator */
            $estimator = new $this->class(...$params);

            $task = new CrossValidate(
                $estimator,
                $dataset,
                $this->validator,
                $this->metric
            );

            $this->backend->enqueue(
                $task,
                [$this, 'afterScore'],
                $estimator->params()
            );
        }

        $scores = $this->backend->process();

        array_multisort($scores, SORT_DESC, $combinations);

        $best = reset($combinations) ?: [];

        $estimator = new $this->base(...array_values($best));

        if ($this->logger) {
            $this->logger->info('Training with best hyper-parameters');
        }

        $estimator->train($dataset);

        $this->base = $estimator;

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->base->predict($dataset);
    }

    /**
     * The callback that executes after the cross validation task.
     *
     * @internal
     *
     * @param float $score
     * @param mixed[] $params
     */
    public function afterScore(float $score, array $params) : void
    {
        if ($this->logger) {
            $this->logger->info("{$this->metric}: $score, "
                . 'params: [' . Params::stringify($params) . ']');
        }
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
        return $this->base->$name(...$arguments);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Grid Search (' . Params::stringify($this->params()) . ')';
    }
}
