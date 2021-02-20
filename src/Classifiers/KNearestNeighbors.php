<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\argmax;
use function array_slice;

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
class KNearestNeighbors implements Estimator, Learner, Online, Probabilistic, Persistable
{
    use AutotrackRevisions;

    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected $k;

    /**
     * Should we consider the distances of our nearest neighbors when making predictions?
     *
     * @var bool
     */
    protected $weighted;

    /**
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The zero vector for the possible class outcomes.
     *
     * @var float[]
     */
    protected $classes = [
        //
    ];

    /**
     * The training samples that make up the neighborhood of the problem space.
     *
     * @var array[]
     */
    protected $samples = [
        //
    ];

    /**
     * The memoized labels of the training set.
     *
     * @var string[]
     */
    protected $labels = [
        //
    ];

    /**
     * @param int $k
     * @param bool $weighted
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $k = 5, bool $weighted = true, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor'
                . " is required to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->kernel = $kernel ?? new Euclidean();
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->samples or !$this->labels) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->samples)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
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
        [$labels, $distances] = $this->nearest($sample);

        if ($this->weighted) {
            $weights = array_fill_keys($labels, 0.0);

            foreach ($distances as $i => $distance) {
                $weights[$labels[$i]] += 1.0 / (1.0 + $distance);
            }
        } else {
            $weights = array_count_values($labels);
        }

        return argmax($weights);
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->samples or !$this->labels or !$this->classes) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->samples)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Predict the probabilities of a single sample and return the joint distribution.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
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
     * Find the K nearest neighbors to the given sample vector using
     * the brute force method.
     *
     * @param list<string|int|float> $sample
     * @return array[]
     */
    protected function nearest(array $sample) : array
    {
        $distances = [];

        foreach ($this->samples as $neighbor) {
            $distances[] = $this->kernel->compute($sample, $neighbor);
        }

        asort($distances);

        $distances = array_slice($distances, 0, $this->k, true);

        $labels = array_intersect_key($this->labels, $distances);

        return [$labels, $distances];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'K Nearest Neighbors (' . Params::stringify($this->params()) . ')';
    }
}
