<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\argmax;

/**
 * K-d Neighbors
 *
 * A fast k nearest neighbors algorithm that uses a binary search tree (BST) to divide the
 * training set into *neighborhoods*. K-d Neighbors then does a binary search to locate the
 * nearest neighborhood of an unknown sample and prunes all neighborhoods whose bounding box
 * is further than the *k*'th nearest neighbor found so far. The main advantage of K-d
 * Neighbors over brute force KNN is that it is much more efficient, however it cannot be
 * partially trained.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDNeighbors implements Estimator, Learner, Probabilistic, Persistable
{
    use AutotrackRevisions;

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
     * The spatial tree used to run nearest neighbor searches.
     *
     * @var Spatial
     */
    protected Spatial $tree;

    /**
     * The zero vector for the possible class outcomes.
     *
     * @var float[]
     */
    protected array $classes = [
        //
    ];

    /**
     * The dimensionality of the training set.
     *
     * @var int|null
     */
    protected ?int $featureCount = null;

    /**
     * @param int $k
     * @param bool $weighted
     * @param Spatial|null $tree
     * @throws InvalidArgumentException
     */
    public function __construct(int $k = 5, bool $weighted = false, ?Spatial $tree = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor'
                . " is required to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->tree = $tree ?? new KDTree();
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
        return $this->tree->kernel()->compatibility();
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
            'tree' => $this->tree,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->tree->bare();
    }

    /**
     * Return the base spatial tree instance.
     *
     * @return Spatial
     */
    public function tree() : Spatial
    {
        return $this->tree;
    }

    /**
     * Train the learner with a dataset.
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

        $this->classes = array_fill_keys($dataset->possibleOutcomes(), 0.0);

        $this->featureCount = $dataset->numFeatures();

        $this->tree->grow($dataset);
    }

    /**
     * Make predictions from a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<string>
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->tree->bare() or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

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
        [$samples, $labels, $distances] = $this->tree->nearest($sample, $this->k);

        if ($this->weighted) {
            $weights = array_fill_keys($labels, 0.0);

            foreach ($labels as $i => $label) {
                $weights[$label] += 1.0 / (1.0 + $distances[$i]);
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
        if ($this->tree->bare() or !$this->classes or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Predict the probabilities of a single sample and return the joint distribution.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        [$samples, $labels, $distances] = $this->tree->nearest($sample, $this->k);

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
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'K-d Neighbors (' . Params::stringify($this->params()) . ')';
    }
}
