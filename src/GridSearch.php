<?php

namespace Rubix\ML;

use Rubix\ML\Helpers\Params;
use Rubix\ML\Backends\Backend;
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
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Generator;
use ReflectionClass;

use function in_array;
use function class_exists;
use function array_unique;
use function array_keys;
use function array_key_exists;
use function is_array;

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
class GridSearch implements EstimatorWrapper, Learner, Parallel, Verbose, Persistable
{
    use AutotrackRevisions, Multiprocessing, LoggerAware;

    /**
     * The threshold for the number of search parameter combinations considered to be huge.
     */
    protected const int HUGE_SPACE_THRESHOLD = 1000;

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
     * @var Metric
     */
    protected Metric $metric;

    /**
     * The validator used to test the estimator.
     *
     * @var Validator
     */
    protected Validator $validator;

    /**
     * The base estimator instance.
     *
     * @var Learner
     */
    protected Learner $base;

    /**
     * The validation scores obtained from the last search.
     *
     * @var list<float>|null
     */
    protected ?array $scores = null;

    /**
     * Return a Grid Search instance from a set of hyper-parameters keyed by the
     * name of the base learner's constructor parameter.
     *
     * @param class-string $class
     * @param array<string, list<mixed>> $params
     * @param Metric|null $metric
     * @param Validator|null $validator
     * @throws InvalidArgumentException
     * @return self
     */
    public static function fromNamedParams(
        string $class,
        array $params,
        ?Metric $metric = null,
        ?Validator $validator = null
    ) : self {
        if (!class_exists($class)) {
            throw new InvalidArgumentException("Class $class does not exist.");
        }

        $reflector = new ReflectionClass($class);

        $parameters = $reflector->getConstructor()?->getParameters() ?? [];

        $names = [];

        foreach ($parameters as $parameter) {
            $names[] = $parameter->getName();
        }

        foreach (array_keys($params) as $name) {
            if (!in_array($name, $names, true)) {
                throw new InvalidArgumentException("$name is not a constructor"
                    . " parameter of $class.");
            }
        }

        $ordered = [];

        foreach ($parameters as $parameter) {
            $name = $parameter->getName();

            if (array_key_exists($name, $params)) {
                $ordered[] = $params[$name];

                continue;
            }

            if ($parameter->isDefaultValueAvailable()) {
                $ordered[] = [$parameter->getDefaultValue()];

                continue;
            }

                throw new InvalidArgumentException("$name is a required constructor"
                    . " parameter of $class.");
        }

        return new self($class, $ordered, $metric, $validator);
    }

    /**
     * Return the names of a class' constructor parameters.
     *
     * @param class-string $class
     * @return list<string>
     */
    protected static function constructorParamNames(string $class) : array
    {
        $reflector = new ReflectionClass($class);

        $constructor = $reflector->getConstructor();

        $names = [];

        if ($constructor) {
            foreach ($constructor->getParameters() as $parameter) {
                $names[] = $parameter->getName();
            }
        }

        return $names;
    }

    /**
     * @param class-string $class
     * @param array<mixed> $params
     * @param Metric|null $metric
     * @param Validator|null $validator
     * @throws InvalidArgumentException
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

        if (!array_is_list($params)) {
            throw new InvalidArgumentException('Hyper-parameters must be'
                . ' supplied in the order they are given to the constructor.');
        }

        foreach ($params as &$tuple) {
            if (!is_array($tuple)) {
                throw new InvalidArgumentException('Each param value must be an array.');
            }

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
            }
        }

        $this->class = $class;
        $this->params = $params;
        $this->metric = $metric;
        $this->validator = $validator ?? new KFold(5);
        $this->base = $proxy;
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
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
     * @return list<DataType>
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
     * Return the parallel processing backend, initializing it with the default if it has
     * not been set yet.
     *
     * @internal
     *
     * @return Backend
     */
    public function backend() : Backend
    {
        return $this->backend ??= new Serial();
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
     * @return Estimator
     */
    public function base() : Estimator
    {
        return $this->base;
    }

    /**
     * Return the validation score for each parameter combination.
     *
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->scores;
    }

    /**
     * Return a table of the validation score obtained from each parameter
     * combination from the last search.
     *
     * @return Generator<mixed[]>
     */
    public function results() : Generator
    {
        if (!$this->scores) {
            return;
        }

        $combinations = $this->combinations();
        $scores = $this->scores;

        array_multisort($scores, SORT_DESC, $combinations);

        $names = self::constructorParamNames($this->class);

        foreach ($scores as $i => $score) {
            $row = [];

            foreach ($combinations[$i] as $j => $param) {
                $row[$names[$j] ?? 'param ' . ($j + 1)] = Params::toString($param);
            }

            $row["{$this->metric}"] = Params::toString($score);

            yield $row;
        }
    }

    /**
     * Return the best combination of parameters found during the last search along
     * with their validation score in a 2-tuple.
     *
     * @return array{0: array<mixed>|null, 1: float|null}
     */
    public function best() : array
    {
        if (!$this->scores) {
            return [null, null];
        }

        $params = iterator_first($this->results());

        $score = array_pop($params);

        return [$params, $score];
    }

    /**
     * Return a list of all possible combinations of parameters i.e their Cartesian product.
     *
     * @return list<list<mixed>>
     */
    public function combinations() : array
    {
        $combinations = [[]];

        /** @var int<0,max> $i */
        foreach ($this->params as $i => $params) {
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
     * Train one estimator per combination of parameters given by the grid and
     * assign the best one as the base estimator of this instance.
     *
     * @param Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        $combinations = $this->combinations();

        if ($this->logger) {
            $this->logger->info("Training $this");

            $numCombinations = number_format(count($combinations));

            $this->logger->info("Total parameter combinations is {$numCombinations}");
        }

        if (count($combinations) > self::HUGE_SPACE_THRESHOLD) {
            warn('Huge search space detected, consider reducing the number of search parameters.');
        }

        $this->backend()->flush();

        foreach ($combinations as $params) {
            /** @var Learner $estimator */
            $estimator = new $this->class(...$params);

            $task = new CrossValidate(
                $estimator,
                $dataset,
                $this->validator,
                $this->metric
            );

            $after = function (float $score) use ($params) {
                if ($this->logger) {
                    $this->logger->info("{$this->metric}: $score, "
                        . 'params: [' . Params::stringify($params) . ']');
                }
            };

            $this->backend()->enqueue($task, $after);
        }

        $scores = $this->backend()->process();

        $this->scores = $scores;

        array_multisort($scores, SORT_DESC, $combinations);

        $best = $combinations[array_key_first($combinations)];

        $estimator = new $this->base(...$best);

        if ($this->logger) {
            $this->logger->info('Now training with best hyper-parameters'
                . Params::stringify($best) . ' on full dataset.');
        }

        $estimator->train($dataset);

        if ($this->logger) {
            $this->logger->info('Training complete');
        }

        $this->base = $estimator;
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param Dataset $dataset
     * @throws Exceptions\RuntimeException
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->base->predict($dataset);
    }

    /**
     * Allow methods to be called on the estimator from the wrapper.
     *
     * @param string $name
     * @param mixed[] $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments) : mixed
    {
        return $this->base->$name(...$arguments);
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['backend']);

        return $properties;
    }

    /**
     * Restore the object from an associative array of serialized properties.
     *
     * @param mixed[] $properties
     */
    public function __unserialize(array $properties) : void
    {
        foreach ($properties as $property => $value) {
            $this->{$property} = $value;
        }
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
