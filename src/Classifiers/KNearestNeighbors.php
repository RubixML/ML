<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Parallel;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Traits\Multiprocessing;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Tasks\Task;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use SplMaxHeap;

use function Rubix\ML\argmax;
use function array_count_values;
use function array_fill_keys;
use function array_map;
use function array_merge;
use function ceil;

/**
 * K Nearest Neighbors
 *
 * A distance-based learning algorithm that locates the *k* nearest samples from the
 * training set and predicts the class label that is most common.
 *
 * > **Note:** This learner is considered a *lazy* learner because it does the majority
 * of its computation during inference. For a fast spatial tree-accelerated version, see
 * KD Neighbors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KNearestNeighbors implements Estimator, Learner, Online, Probabilistic, Parallel, Persistable
{
    use Multiprocessing, AutotrackRevisions;

    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected int $k;

    /**
     * Should we consider the distances of our nearest neighbors when making predictions?
     *
     * @var bool
     */
    protected bool $weighted;

    /**
     * The distance function to use when computing the distances.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The zero vector for the possible class outcomes.
     *
     * @var float[]
     */
    protected array $classes = [
        //
    ];

    /**
     * The training samples.
     *
     * @var list<list<string|int|float>>
     */
    protected array $samples = [
        //
    ];

    /**
     * The training labels.
     *
     * @var string[]
     */
    protected array $labels = [
        //
    ];

    /**
     * Infer a chunk of samples.
     *
     * @internal
     *
     * @param self $estimator
     * @param Dataset $chunk
     * @return array
     */
    public static function predictChunk(self $estimator, Dataset $chunk) : array
    {
        return array_map([$estimator, 'predictSample'], $chunk->samples());
    }

    /**
     * Estimate the joint probabilities for each possible outcome in a chunk of samples.
     *
     * @internal
     *
     * @param self $estimator
     * @param Dataset $chunk
     * @return list<array<string,float>>
     */
    public static function probaChunk(self $estimator, Dataset $chunk) : array
    {
        return array_map([$estimator, 'probaSample'], $chunk->samples());
    }

    /**
     * @param int $k
     * @param bool $weighted
     * @param Distance|null $kernel
     * @throws InvalidArgumentException
     */
    public function __construct(int $k = 5, bool $weighted = false, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor'
                . " is required to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->kernel = $kernel ?? new Euclidean();
        $this->backend = new Serial();
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
        return EstimatorType::classifier();
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
        return $this->kernel->compatibility();
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
            'k' => $this->k,
            'weighted' => $this->weighted,
            'kernel' => $this->kernel,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->samples and $this->labels;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        $this->samples = $this->labels = [];

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        foreach ($dataset->possibleOutcomes() as $class) {
            if (!isset($this->classes[$class])) {
                $this->classes[$class] = 0.0;
            }
        }

        $this->samples = array_merge($this->samples, $dataset->samples());
        $this->labels = array_merge($this->labels, $dataset->labels());
    }

    /**
     * Make predictions from a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->samples or !$this->labels) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->samples)))->check();

        $chunkSize = (int) ceil($dataset->numSamples() / $this->backend->workers());

        $this->backend->flush();

        foreach ($dataset->batch($chunkSize) as $chunk) {
            $task = new Task([self::class, 'predictChunk'], [$this, $chunk]);

            $this->backend->enqueue($task);
        }

        $predictions = [];

        foreach ($this->backend->process() as $output) {
            /** @var list<string> $output */
            $predictions = array_merge($predictions, $output);
        }

        return $predictions;
    }

    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return string
     */
    public function predictSample(array $sample) : string
    {
        /** @var array<string> $labels */
        [$labels, $distances] = $this->nearest($sample);

        if ($this->weighted) {
            $weights = array_fill_keys($labels, 0.0);

            foreach ($distances as $i => $distance) {
                $weights[$labels[$i]] += 1.0 / (1.0 + $distance);
            }
        } else {
            $weights = array_count_values($labels);
        }

        /** @var array<string,float|int> $weights */
        return argmax($weights);
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<array<string,float>>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->samples or !$this->labels or !$this->classes) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->samples)))->check();

        $chunkSize = (int) ceil($dataset->numSamples() / $this->backend->workers());

        $this->backend->flush();

        foreach ($dataset->batch($chunkSize) as $chunk) {
            $task = new Task([self::class, 'probaChunk'], [$this, $chunk]);

            $this->backend->enqueue($task);
        }

        $probabilities = [];

        foreach ($this->backend->process() as $output) {
            /** @var list<array<string,float>> $output */
            $probabilities = array_merge($probabilities, $output);
        }

        return $probabilities;
    }

    /**
     * Predict the probabilities of a single sample and return the joint distribution.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return array<string,float>
     */
    public function probaSample(array $sample) : array
    {
        /** @var array<string> $labels */
        [$labels, $distances] = $this->nearest($sample);

        if ($this->weighted) {
            $weights = array_fill_keys($labels, 0.0);

            foreach ($labels as $i => $label) {
                $weights[$label] += 1.0 / (1.0 + $distances[$i]);
            }
        } else {
            $weights = array_count_values($labels);
        }

        $total = array_sum($weights);

        $dist = $this->classes;

        foreach ($weights as $class => $weight) {
            $dist[$class] = $weight / $total;
        }

        return $dist;
    }

    /**
     * Find the K nearest neighbors to the given sample vector using the brute force method.
     *
     * @param list<string|int|float> $sample
     * @return array{list<string|int|float>,list<float>}
     */
    protected function nearest(array $sample) : array
    {
        $heap = new SplMaxHeap();

        foreach ($this->samples as $index => $neighbor) {
            $distance = $this->kernel->compute($sample, $neighbor);

            if (is_nan($distance)) {
                continue;
            }

            if ($heap->count() < $this->k) {
                $heap->insert([$distance, $index]);

                continue;
            }

            if ($distance >= $heap->top()[0]) {
                continue;
            }

            $heap->extract();

            $heap->insert([$distance, $index]);
        }

        $labels = $distances = [];

        foreach ($heap as [$distance, $index]) {
            $labels[] = $this->labels[$index];

            $distances[] = $distance;
        }

        return [$labels, $distances];
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
        return 'K Nearest Neighbors (' . Params::stringify($this->params()) . ')';
    }
}
