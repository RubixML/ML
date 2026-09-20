<?php

namespace Rubix\ML;

use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Validator;
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
use ReflectionClass;
use Generator;

use function in_array;
use function class_exists;
use function array_keys;
use function array_pop;
use function array_map;
use function array_sum;
use function array_slice;
use function array_column;
use function array_unique;
use function array_fill_keys;
use function array_multisort;
use function array_key_exists;
use function array_is_list;
use function array_key_last;
use function is_array;
use function count;
use function min;
use function max;
use function floor;
use function serialize;
use function rand;
use function getrandmax;
use function number_format;

/**
 * Bayesian Search
 *
 * Bayesian Search is a form of hyper-parameter optimization that uses the Tree-structured
 * Parzen Estimator (TPE) to iteratively propose the next set of hyper-parameters to
 * evaluate based on the results of the trials before it. From the user's perspective, the
 * process of training and predicting is the same, however, under the hood Bayesian Search
 * evaluates a sequence of trials and selects the best performing hyper-parameters as the
 * base estimator.
 *
 * > **Note:** The candidate search space is declared by listing the possible values of
 * each of the base learner's constructor parameters. Bayesian Search samples the space
 * one candidate combination at a time, updating its beliefs after every trial, and trains
 * at most *max trials* candidate models per training session.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BayesianSearch implements EstimatorWrapper, Learner, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The threshold for the number of possible parameter combinations considered to be huge.
     */
    protected const int HUGE_SEARCH_THRESHOLD = 1000;

    /**
     * The number of times to resample a candidate before giving up and returning the best
     * hyper-parameters found so far.
     */
    protected const int MAX_RESAMPLES = 10;

    /**
     * The Laplace smoothing factor applied to the likelihood ratio of each candidate value.
     */
    protected const float SMOOTHING = 1.0;

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
     * The maximum number of trials to run during a single training session.
     *
     * @var int
     */
    protected int $maxTrials;

    /**
     * The fraction of the top performing trials to use as the *good* set during sampling.
     *
     * @var float
     */
    protected float $quantile;

    /**
     * The number of initial trials to run using random search before beginning estimation.
     *
     * @var int
     */
    protected int $startup;

    /**
     * The hyper-parameters evaluated during the last search along with their validation scores.
     *
     * @var list<array{params: list<mixed>, score: float}>
     */
    protected array $history = [
        //
    ];

    /**
     * Return a Bayesian Search instance from a set of hyper-parameters keyed by the
     * name of the base learner's constructor parameter.
     *
     * @param class-string $class
     * @param array<string, list<mixed>> $params
     * @param Metric|null $metric
     * @param Validator|null $validator
     * @param int $maxTrials
     * @param float $quantile
     * @param int $startup
     * @throws InvalidArgumentException
     * @return self
     */
    public static function fromNamedParams(
        string $class,
        array $params,
        ?Metric $metric = null,
        ?Validator $validator = null,
        int $maxTrials = 32,
        float $quantile = 0.25,
        int $startup = 10
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

        return new self($class, $ordered, $metric, $validator, $maxTrials, $quantile, $startup);
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
     * @param int $maxTrials
     * @param float $quantile
     * @param int $startup
     * @throws InvalidArgumentException
     */
    public function __construct(
        string $class,
        array $params,
        ?Metric $metric = null,
        ?Validator $validator = null,
        int $maxTrials = 32,
        float $quantile = 0.25,
        int $startup = 10
    ) {
        if (!class_exists($class)) {
            throw new InvalidArgumentException("Class $class does not exist.");
        }

        if (!array_is_list($params)) {
            throw new InvalidArgumentException('Hyper-parameters must be'
                . ' supplied in the order they are given to the constructor.');
        }

        if ($maxTrials < 1) {
            throw new InvalidArgumentException('Maximum number of trials'
                . ' must be greater than 0.');
        }

        if ($quantile <= 0.0 or $quantile >= 1.0) {
            throw new InvalidArgumentException('Quantile must be between'
                . ' 0 and 1 exclusive.');
        }

        if ($startup < 0) {
            throw new InvalidArgumentException('Number of startup trials'
                . ' must be greater than or equal to 0.');
        }

        $reflector = new ReflectionClass($class);

        $parameters = $reflector->getConstructor()?->getParameters() ?? [];

        foreach ($params as $index => &$tuple) {
            if (!is_array($tuple)) {
                throw new InvalidArgumentException('Each param value must be an array.');
            }

            if (empty($tuple)) {
                $tuple = [null];

                $parameter = $parameters[$index] ?? null;

                if ($parameter and $parameter->isDefaultValueAvailable()) {
                    $tuple = [$parameter->getDefaultValue()];
                }
            } else {
                $tuple = array_unique($tuple, SORT_REGULAR);
            }
        }

        $proxy = new $class(...array_map('current', $params));

        if (!$proxy instanceof Learner) {
            throw new InvalidArgumentException('Base class must'
                . ' implement the Learner Interface.');
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
        $this->maxTrials = $maxTrials;
        $this->quantile = $quantile;
        $this->startup = $startup;
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
            'maxTrials' => $this->maxTrials,
            'quantile' => $this->quantile,
            'startup' => $this->startup,
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
     * @return Estimator
     */
    public function base() : Estimator
    {
        return $this->base;
    }

    /**
     * Return the validation score for each trial from the last search in the order they
     * were evaluated.
     *
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->history ? array_column($this->history, 'score') : null;
    }

    /**
     * Return a table of the hyper-parameters evaluated during the last search along with
     * their validation scores, sorted by score descending.
     *
     * @return Generator<mixed[]>
     */
    public function results() : Generator
    {
        if (!$this->history) {
            return;
        }

        $scores = array_column($this->history, 'score');
        $trialParams = array_column($this->history, 'params');

        array_multisort($scores, SORT_DESC, $trialParams);

        $names = self::constructorParamNames($this->class);

        foreach ($trialParams as $i => $params) {
            $row = [];

            foreach ($params as $j => $param) {
                $row[$names[$j] ?? 'param ' . ($j + 1)] = Params::toString($param);
            }

            $row["{$this->metric}"] = Params::toString($scores[$i]);

            yield $row;
        }
    }

    /**
     * Return the best combination of parameters found during the last search along with
     * their validation score in a 2-tuple.
     *
     * @return array{0: array<mixed>|null, 1: float|null}
     */
    public function best() : array
    {
        if (!$this->history) {
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
     * Train one estimator per trial using the TPE sampler to propose hyper-parameters and
     * assign the best performing one as the base estimator of this instance.
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

        $spaceSize = count($this->combinations());

        $maxTrials = min($this->maxTrials, $spaceSize);

        if ($this->logger) {
            $this->logger->info("Training $this");

            $numTrials = number_format($maxTrials);

            $this->logger->info("Total number of trials is {$numTrials}");
        }

        if ($spaceSize > self::HUGE_SEARCH_THRESHOLD) {
            warn('Huge search space detected, consider reducing the number of search parameters.');
        }

        $history = [];

        while (count($history) < $maxTrials) {
            $params = $this->propose($history);

            /** @var Learner $estimator */
            $estimator = new $this->class(...$params);

            $score = $this->validator->test($estimator, $dataset, $this->metric);

            if ($this->logger) {
                $this->logger->info("{$this->metric}: $score, "
                    . 'params: [' . Params::stringify($params) . ']');
            }

            $history[] = [
                'params' => $params,
                'score' => $score,
            ];
        }

        $this->history = $history;

        $params = $this->bestParams($history);

        $estimator = new $this->base(...$params);

        if ($this->logger) {
            $this->logger->info('Training with best hyper-parameters'
                . Params::stringify($params) . ' on full dataset.');
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
     * Propose the next combination of hyper-parameters using random search during the
     * startup phase and the TPE sampler thereafter.
     *
     * @param list<array{params: list<mixed>, score: float}> $history
     * @return list<mixed>
     */
    protected function propose(array $history) : array
    {
        $evaluated = [];

        foreach ($history as $trial) {
            $evaluated[serialize($trial['params'])] = true;
        }

        for ($attempt = 0; $attempt < self::MAX_RESAMPLES; ++$attempt) {
            if (count($history) < $this->startup) {
                $candidate = $this->sample();
            } else {
                $candidate = $this->tpe($history);
            }

            if (!isset($evaluated[serialize($candidate)])) {
                return $candidate;
            }
        }

        foreach ($this->combinations() as $candidate) {
            if (!isset($evaluated[serialize($candidate)])) {
                return $candidate;
            }
        }

        return $this->bestParams($history);

    /**
     * Sample a random combination of hyper-parameters uniformly from the search space.
     *
     * @return list<mixed>
     */
    protected function sample() : array
    {
        $candidate = [];

        foreach ($this->params as $params) {
            $candidate[] = $params[rand(0, count($params) - 1)];
        }

        return $candidate;
    }

    /**
     * Propose a combination of hyper-parameters using the Tree-structured Parzen Estimator.
     *
     * @param list<array{params: list<mixed>, score: float}> $history
     * @return list<mixed>
     */
    protected function tpe(array $history) : array
    {
        $scores = array_column($history, 'score');
        $trialParams = array_column($history, 'params');

        array_multisort($scores, SORT_DESC, $trialParams);

        $top = max(1, (int) floor($this->quantile * count($history)));

        $good = array_slice($trialParams, 0, $top);

        $bad = array_slice($trialParams, $top);

        if (empty($bad)) {
            return $this->sample();
        }

        $candidate = [];

        foreach ($this->params as $i => $params) {
            $keys = array_map('serialize', $params);

            $goodCounts = array_fill_keys($keys, 0);

            $badCounts = array_fill_keys($keys, 0);

            foreach ($good as $trial) {
                ++$goodCounts[serialize($trial[$i])];
            }

            foreach ($bad as $trial) {
                ++$badCounts[serialize($trial[$i])];
            }

            $weights = [];

            foreach ($keys as $j => $key) {
                $weights[] = ($goodCounts[$key] + self::SMOOTHING)
                    / ($badCounts[$key] + self::SMOOTHING);
            }

            $candidate[] = $this->draw($params, $weights);
        }

        return $candidate;
    }

    /**
     * Draw a value from a finite set of candidates with probability proportional to the
     * given weights.
     *
     * @param list<mixed> $values
     * @param list<float> $weights
     * @return mixed
     */
    protected function draw(array $values, array $weights) : mixed
    {
        $total = array_sum($weights);

        $threshold = $total * rand() / getrandmax();

        foreach ($values as $i => $value) {
            $threshold -= $weights[$i];

            if ($threshold <= 0.0) {
                return $value;
            }
        }

        return $values[array_key_last($values)];
    }

    /**
     * Return the best performing combination of hyper-parameters from a set of trials.
     *
     * @param list<array{params: list<mixed>, score: float}> $history
     * @return list<mixed>
     */
    protected function bestParams(array $history) : array
    {
        $scores = array_column($history, 'score');

        $trialParams = array_column($history, 'params');

        array_multisort($scores, SORT_DESC, $trialParams);

        return $trialParams[0];
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
        return get_object_vars($this);
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
        return 'Bayesian Search (' . Params::stringify($this->params()) . ')';
    }
}
