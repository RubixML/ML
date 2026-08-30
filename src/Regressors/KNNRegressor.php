<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Parallel;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Traits\Multiprocessing;
use Rubix\ML\Backends\Backend;
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

/**
 * KNN Regressor
 *
 * A version of the K Nearest Neighbors algorithm that uses the average (mean) outcome of
 * the *k* nearest data points to an unknown sample in order to make continuous-valued
 * predictions suitable for regression problems.
 *
 * > **Note:** This learner is considered a *lazy* learner because it does the majority
 * of its computation during inference. For a fast spatial tree-accelerated version, see
 * KD Neighbors Regressor.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KNNRegressor implements Estimator, Learner, Online, Parallel, Persistable
{
    use AutotrackRevisions;
    use Multiprocessing;

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
     * The distance kernel to use when computing the distances.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The training samples.
     *
     * @var list<(string|int|float)[]>
     */
    protected array $samples = [
        //
    ];

    /**
     * The training labels.
     *
     * @var list<int|float>
     */
    protected array $labels = [
        //
    ];

    /**
     * @param int $k
     * @param bool $weighted
     * @param Distance|null $kernel
     * @throws InvalidArgumentException
     */
    public function __construct(int $k = 5, bool $weighted = false, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * Make predictions on a chunk of samples.
     *
     * @internal
     *
     * @param Dataset $chunk
     * @return list<int|float>
     */
    public function predictChunk(Dataset $chunk) : array
    {
        return array_map([$this, 'predictSample'], $chunk->samples());
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
        return EstimatorType::regressor();
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

        $this->samples = array_merge($this->samples, $dataset->samples());
        $this->labels = array_merge($this->labels, $dataset->labels());
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int|float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->samples or !$this->labels) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->samples)))->check();

        $chunkSize = (int) ceil($dataset->numSamples() / $this->backend()->workers());

        $this->backend()->flush();

        foreach ($dataset->batch($chunkSize) as $chunk) {
            $task = new Task([$this, 'predictChunk'], [$chunk]);

            $this->backend()->enqueue($task);
        }

        $predictions = [];

        foreach ($this->backend()->process() as $output) {
            /** @var list<int|float> $output */
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
     * @return int|float
     */
    public function predictSample(array $sample) : int|float
    {
        [$labels, $distances] = $this->nearest($sample);

        if ($this->weighted) {
            $weights = [];

            foreach ($distances as $i => $distance) {
                $weights[$i] = 1.0 / (1.0 + $distance);
            }

            return Stats::weightedMean($labels, $weights);
        }

        return Stats::mean($labels);
    }

    /**
     * Return the parallel processing backend, initializing it with the default if it has
     * not been set yet.
     *
     * @return Backend
     */
    protected function backend() : Backend
    {
        return $this->backend ??= new Serial();
    }

    /**
     * Find the K nearest neighbors to the given sample vector using the brute force method.
     *
     * @param list<string|int|float> $sample
     * @return array{list<int|float|string>,list<float>}
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
        return 'KNN Regressor (' . Params::stringify($this->params()) . ')';
    }
}
