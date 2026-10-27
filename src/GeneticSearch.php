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
use Rubix\ML\Exceptions\RuntimeException;
use ReflectionClass;
use Throwable;
use Closure;

use function in_array;
use function class_exists;
use function array_keys;
use function array_map;
use function array_column;
use function array_key_exists;
use function array_is_list;
use function array_fill;
use function arsort;
use function call_user_func;
use function count;
use function get_object_vars;
use function getrandmax;
use function implode;
use function intdiv;
use function is_array;
use function is_finite;
use function is_resource;
use function min;
use function number_format;
use function rand;
use function serialize;

/**
 * Genetic Search
 *
 * Genetic Search optimizes a base learner's hyper-parameters using a population of
 * candidate combinations. Each generation retains its best candidates and creates
 * new ones through tournament selection, uniform crossover, and mutation. Each
 * distinct candidate is cross-validated at most once, and the best combination is
 * trained on the full dataset after the search.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */
class GeneticSearch implements Estimator, Learner, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The maximum number of attempts to draw an unseen candidate before scanning
     * the finite search space.
     */
    protected const int MAX_RESAMPLES = 32;

    /**
     * The class name of the base estimator.
     *
     * @var class-string
     */
    protected string $class;

    /**
     * The possible values for each base learner constructor parameter.
     *
     * @var list<list<mixed>>
     */
    protected array $params;

    /**
     * The number of candidates in each generation.
     */
    protected int $populationSize;

    /**
     * The maximum number of generations, including the initial population.
     */
    protected int $generations;

    /**
     * The probability of crossing two parents to create an offspring.
     */
    protected float $crossoverRate;

    /**
     * The probability of changing each gene in an offspring.
     */
    protected float $mutationRate;

    /**
     * The number of best candidates carried into the next generation.
     */
    protected int $eliteCount;

    /**
     * The validation metric used to score each candidate.
     */
    protected Metric $metric;

    /**
     * The validator used to test each candidate.
     */
    protected Validator $validator;

    /**
     * The base estimator instance.
     *
     * @var Learner & Estimator
     */
    protected Learner & Estimator $base;

    /**
     * The unique candidates evaluated during the last search in trial order.
     *
     * @var list<array{params: list<mixed>, score: float}>
     */
    protected array $history = [];

    /**
     * A callback invoked on each estimator before cross-validation.
     *
     * @var Closure(Learner & Estimator) : mixed|null
     */
    protected ?Closure $setup = null;

    /**
     * Return a Genetic Search instance from hyper-parameters keyed by the base
     * learner's constructor parameter names.
     *
     * @param class-string $class
     * @param array<string, list<mixed>> $params
     * @param int $populationSize
     * @param int $generations
     * @param float $crossoverRate
     * @param float $mutationRate
     * @param int $eliteCount
     * @param Metric|null $metric
     * @param Validator|null $validator
     * @throws InvalidArgumentException
     * @return self
     */
    public static function fromNamedParams(
        string $class,
        array $params,
        int $populationSize = 16,
        int $generations = 10,
        float $crossoverRate = 0.8,
        float $mutationRate = 0.1,
        int $eliteCount = 1,
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

        return new self(
            $class,
            $ordered,
            $populationSize,
            $generations,
            $crossoverRate,
            $mutationRate,
            $eliteCount,
            $metric,
            $validator
        );
    }

    /**
     * Return the names of a class's constructor parameters.
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
     * @param int $populationSize
     * @param int $generations
     * @param float $crossoverRate
     * @param float $mutationRate
     * @param int $eliteCount
     * @param Metric|null $metric
     * @param Validator|null $validator
     * @throws InvalidArgumentException
     */
    public function __construct(
        string $class,
        array $params,
        int $populationSize = 16,
        int $generations = 10,
        float $crossoverRate = 0.8,
        float $mutationRate = 0.1,
        int $eliteCount = 1,
        ?Metric $metric = null,
        ?Validator $validator = null
    ) {
        if (!class_exists($class)) {
            throw new InvalidArgumentException("Class $class does not exist.");
        }

        if (!array_is_list($params)) {
            throw new InvalidArgumentException('Hyper-parameters must be'
                . ' supplied in the order they are given to the constructor.');
        }

        if ($populationSize < 2) {
            throw new InvalidArgumentException('Population size must be at least 2.');
        }

        if ($generations < 1) {
            throw new InvalidArgumentException('Number of generations must be at least 1.');
        }

        if (!is_finite($crossoverRate) or $crossoverRate < 0.0 or $crossoverRate > 1.0) {
            throw new InvalidArgumentException('Crossover rate must be between 0 and 1.');
        }

        if (!is_finite($mutationRate) or $mutationRate < 0.0 or $mutationRate > 1.0) {
            throw new InvalidArgumentException('Mutation rate must be between 0 and 1.');
        }

        if ($eliteCount < 0 or $eliteCount >= $populationSize) {
            throw new InvalidArgumentException('Elite count must be nonnegative and'
                . ' less than population size.');
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
            }

            $unique = [];
            $seen = [];

            foreach ($tuple as $value) {
                if (is_resource($value)) {
                    throw new InvalidArgumentException(
                        'Hyper-parameter values must not be resources.'
                    );
                }

                try {
                    $key = serialize($value);
                } catch (Throwable $exception) {
                    throw new InvalidArgumentException(
                        'Hyper-parameter values must be serializable.',
                        previous: $exception
                    );
                }

                if (!isset($seen[$key])) {
                    $seen[$key] = true;
                    $unique[] = $value;
                }
            }

            $tuple = $unique;
        }

        $proxy = new $class(...array_map('current', $params));

        if (!$proxy instanceof Learner or !$proxy instanceof Estimator) {
            throw new InvalidArgumentException('Base class must'
                . ' implement the Learner and Estimator Interfaces.');
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
        $this->populationSize = $populationSize;
        $this->generations = $generations;
        $this->crossoverRate = $crossoverRate;
        $this->mutationRate = $mutationRate;
        $this->eliteCount = $eliteCount;
        $this->metric = $metric;
        $this->validator = $validator ?? new KFold(5);
        $this->base = $proxy;
    }

    /**
     * Register a callback to configure each new estimator before validation and
     * the best estimator before final training.
     *
     * @param Closure(Learner & Estimator) : mixed $setup
     */
    public function setup(Closure $setup) : void
    {
        $this->setup = $setup;
    }

    /**
     * Return the estimator type.
     *
     * @internal
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
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Return the hyper-parameter settings of the search.
     *
     * @internal
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'class' => $this->class,
            'params' => $this->params,
            'populationSize' => $this->populationSize,
            'generations' => $this->generations,
            'crossoverRate' => $this->crossoverRate,
            'mutationRate' => $this->mutationRate,
            'eliteCount' => $this->eliteCount,
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
     * @return Estimator
     */
    public function base() : Estimator
    {
        return $this->base;
    }

    /**
     * Return validation scores for unique trials in evaluation order.
     *
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->history ? array_column($this->history, 'score') : null;
    }

    /**
     * Return a table of unique trials and their validation scores.
     *
     * @throws RuntimeException
     * @return Report
     */
    public function results() : Report
    {
        if (!$this->history) {
            throw new RuntimeException('No trials have been run yet.');
        }

        $names = self::constructorParamNames($this->class);

        $results = [];

        foreach ($this->history as $i => $trial) {
            $row = [];

            $row["{$this->metric}"] = Params::toString($trial['score']);

            $params = [];

            foreach ($trial['params'] as $j => $param) {
                $params[$names[$j] ?? 'param ' . ($j + 1)] = Params::toString($param);
            }

            $row['params'] = $params;

            $results['Trial ' . ($i + 1)] = $row;
        }

        return new Report($results);
    }

    /**
     * Search the finite parameter space and train the best candidate on the full dataset.
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

        $spaceSize = $this->spaceSize();

        $targetSize = min($this->populationSize, $spaceSize);

        if ($this->logger) {
            $this->logger->info("Training $this");

            $this->logger->info('Population size is ' . number_format($targetSize)
                . ' for up to ' . number_format($this->generations) . ' generations.');
        }

        $seen = [];

        $cursor = array_fill(0, count($this->params), 0);

        $population = [];

        while (count($population) < $targetSize) {
            $candidate = $this->novel(null, $seen, $cursor);

            if ($candidate === null) {
                break;
            }

            $seen[$this->key($candidate)] = true;

            $population[] = $candidate;
        }

        $this->history = [];

        $scores = [];

        for ($generation = 0; $generation < $this->generations; ++$generation) {
            $scored = [];

            foreach ($population as $chromosome) {
                $key = $this->key($chromosome);

                if (!array_key_exists($key, $scores)) {
                    $params = $this->decode($chromosome);

                    /** @var Learner & Estimator $estimator */
                    $estimator = new $this->class(...$params);

                    if ($this->setup) {
                        call_user_func($this->setup, $estimator);
                    }

                    $score = $this->validator->test($estimator, $dataset, $this->metric);

                    $scores[$key] = $score;

                    $this->history[] = [
                        'params' => $params,
                        'score' => $score,
                    ];

                    if ($this->logger) {
                        $trial = count($this->history);

                        $this->logger->info("Trial {$trial}: {$this->metric}: $score, "
                            . 'params: [' . Params::stringify($this->named($params)) . ']');
                    }
                }

                $scored[] = [
                    'chromosome' => $chromosome,
                    'score' => $scores[$key],
                ];
            }

            if ($generation + 1 === $this->generations or count($seen) >= $spaceSize) {
                break;
            }

            $ranked = $this->rank($scored);

            $next = [];

            for ($i = 0; $i < min($this->eliteCount, count($ranked)); ++$i) {
                $next[] = $ranked[$i]['chromosome'];
            }

            while (count($next) < $targetSize and count($seen) < $spaceSize) {
                $candidate = $this->novel($this->breed($scored), $seen, $cursor);

                if ($candidate === null) {
                    break;
                }

                $seen[$this->key($candidate)] = true;

                $next[] = $candidate;
            }

            if (count($next) <= $this->eliteCount) {
                break;
            }

            $population = $next;
        }

        $best = $this->bestParams($this->history);

        /** @var Learner & Estimator $estimator */
        $estimator = new $this->class(...$best);

        if ($this->setup) {
            call_user_func($this->setup, $estimator);
        }

        if ($this->logger) {
            $this->logger->info('Training with best hyper-parameters '
                . Params::stringify($this->named($best)) . ' on full dataset.');
        }

        $estimator->train($dataset);

        if ($this->logger) {
            $this->logger->info('Training complete');
        }

        $this->base = $estimator;
    }

    /**
     * Make predictions on a dataset with the trained base estimator.
     *
     * @param Dataset $dataset
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->base->predict($dataset);
    }

    /**
     * Return the number of combinations, saturating at PHP_INT_MAX.
     *
     * @return int
     */
    protected function spaceSize() : int
    {
        $size = 1;

        foreach ($this->params as $values) {
            $count = count($values);

            if ($size > intdiv(PHP_INT_MAX, $count)) {
                return PHP_INT_MAX;
            }

            $size *= $count;
        }

        return $size;
    }

    /**
     * Return the key for a chromosome of parameter-value indexes.
     *
     * @param list<int> $chromosome
     * @return string
     */
    protected function key(array $chromosome) : string
    {
        return implode(':', $chromosome);
    }

    /**
     * Decode parameter-value indexes to constructor arguments.
     *
     * @param list<int> $chromosome
     * @return list<mixed>
     */
    protected function decode(array $chromosome) : array
    {
        $params = [];

        foreach ($chromosome as $i => $choice) {
            $params[] = $this->params[$i][$choice];
        }

        return $params;
    }

    /**
     * Return a uniformly sampled chromosome.
     *
     * @return list<int>
     */
    protected function sample() : array
    {
        $chromosome = [];

        foreach ($this->params as $values) {
            $chromosome[] = rand(0, count($values) - 1);
        }

        return $chromosome;
    }

    /**
     * Find a candidate that has not already been reserved or evaluated.
     *
     * @param list<int>|null $proposal
     * @param array<string, bool> $seen
     * @param list<int> $cursor
     * @return list<int>|null
     */
    protected function novel(?array $proposal, array $seen, array &$cursor) : ?array
    {
        if ($proposal !== null and !isset($seen[$this->key($proposal)])) {
            return $proposal;
        }

        for ($attempt = 0; $attempt < self::MAX_RESAMPLES; ++$attempt) {
            $candidate = $this->sample();

            if (!isset($seen[$this->key($candidate)])) {
                return $candidate;
            }
        }

        $start = $cursor;

        do {
            $candidate = $cursor;

            $cursor = $this->increment($cursor);

            if (!isset($seen[$this->key($candidate)])) {
                return $candidate;
            }
        } while ($cursor !== $start);

        return null;
    }

    /**
     * Advance a chromosome in mixed-radix order, wrapping at the end.
     *
     * @param list<int> $chromosome
     * @return list<int>
     */
    protected function increment(array $chromosome) : array
    {
        for ($i = count($chromosome) - 1; $i >= 0; --$i) {
            ++$chromosome[$i];

            if ($chromosome[$i] < count($this->params[$i])) {
                return $chromosome;
            }

            $chromosome[$i] = 0;
        }

        return $chromosome;
    }

    /**
     * Rank scored candidates by descending validation score.
     *
     * @param list<array{chromosome: list<int>, score: float}> $population
     * @return list<array{chromosome: list<int>, score: float}>
     */
    protected function rank(array $population) : array
    {
        $scores = array_column($population, 'score');

        arsort($scores, SORT_NUMERIC);

        $ranked = [];

        foreach (array_keys($scores) as $index) {
            $ranked[] = $population[$index];
        }

        return $ranked;
    }

    /**
     * Choose the fitter of two uniformly drawn members of the population.
     *
     * @param list<array{chromosome: list<int>, score: float}> $population
     * @return list<int>
     */
    protected function tournament(array $population) : array
    {
        $first = $population[rand(0, count($population) - 1)];

        $second = $population[rand(0, count($population) - 1)];

        return $first['score'] >= $second['score']
            ? $first['chromosome']
            : $second['chromosome'];
    }

    /**
     * Create offspring using uniform crossover and independent gene mutation.
     *
     * @param list<array{chromosome: list<int>, score: float}> $population
     * @return list<int>
     */
    protected function breed(array $population) : array
    {
        $first = $this->tournament($population);

        $second = $this->tournament($population);

        $offspring = $first;

        if ($this->randomUnit() < $this->crossoverRate) {
            foreach ($offspring as $i => $gene) {
                if (rand(0, 1) === 1) {
                    $offspring[$i] = $second[$i];
                }
            }
        }

        foreach ($offspring as $i => $gene) {
            $choices = count($this->params[$i]);

            if ($choices > 1 and $this->randomUnit() < $this->mutationRate) {
                $replacement = rand(0, $choices - 2);

                if ($replacement >= $gene) {
                    ++$replacement;
                }

                $offspring[$i] = $replacement;
            }
        }

        return $offspring;
    }

    /**
     * Draw a pseudo-random value from the half-open interval [0, 1).
     *
     * @return float
     */
    protected function randomUnit() : float
    {
        return rand() / (getrandmax() + 1.0);
    }

    /**
     * Return the best parameter combination from the completed trials.
     *
     * @param list<array{params: list<mixed>, score: float}> $history
     * @return list<mixed>
     */
    protected function bestParams(array $history) : array
    {
        $best = $history[0];

        foreach ($history as $trial) {
            if ($trial['score'] > $best['score']) {
                $best = $trial;
            }
        }

        return $best['params'];
    }

    /**
     * Reindex positional parameters by their constructor names.
     *
     * @internal
     * @param list<mixed> $params
     * @return mixed[]
     */
    protected function named(array $params) : array
    {
        $names = self::constructorParamNames($this->class);

        $named = [];

        foreach ($params as $i => $param) {
            $named[$names[$i] ?? 'param ' . ($i + 1)] = $param;
        }

        return $named;
    }

    /**
     * Forward unknown method calls to the trained base estimator.
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
     * Return properties used to serialize the search, excluding the setup callback.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['setup']);

        return $properties;
    }

    /**
     * Restore serialized properties.
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
     * Return the string representation of the search.
     *
     * @internal
     * @return string
     */
    public function __toString() : string
    {
        return 'Genetic Search (' . Params::stringify($this->params()) . ')';
    }
}
