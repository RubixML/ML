<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BallTree;
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
use function in_array;

/**
 * Radius Neighbors
 *
 * Radius Neighbors is a spatial tree-based classifier that takes the weighted vote of each
 * neighbor within a fixed user-defined radius. Since the radius of the search can be
 * constrained, Radius Neighbors is more robust to outliers than K Nearest Neighbors. In
 * addition, Radius Neighbors acts as a quasi outlier detector by flagging samples that have
 * 0 neighbors within radius.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RadiusNeighbors implements Estimator, Learner, Probabilistic, Persistable
{
    use AutotrackRevisions;

    /**
     * The radius within which points are considered neighbors.
     *
     * @var float
     */
    protected float $radius;

    /**
     * Should we consider the distances of our nearest neighbors when making predictions?
     *
     * @var bool
     */
    protected bool $weighted;

    /**
     * The spatial tree used to run range searches.
     *
     * @var Spatial
     */
    protected Spatial $tree;

    /**
     * The class label for any samples that have 0 neighbors within the specified radius.
     *
     * @var string
     */
    protected string $outlierClass;

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
     * @param float $radius
     * @param bool $weighted
     * @param string $outlierClass
     * @param Spatial|null $tree
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $radius = 1.0,
        bool $weighted = false,
        string $outlierClass = '?',
        ?Spatial $tree = null
    ) {
        if ($radius <= 0.0) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        if (empty($outlierClass)) {
            throw new InvalidArgumentException('Anomaly class'
                . ' cannot be an empty string.');
        }

        $this->radius = $radius;
        $this->weighted = $weighted;
        $this->outlierClass = $outlierClass;
        $this->tree = $tree ?? new BallTree();
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
            'radius' => $this->radius,
            'weighted' => $this->weighted,
            'outlier class' => $this->outlierClass,
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
     * @throws RuntimeException
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        $classes = $dataset->possibleOutcomes();

        if (in_array($this->outlierClass, $classes)) {
            throw new RuntimeException('Training set cannot contain'
                . ' labels of the outlier class.');
        }

        $classes[] = $this->outlierClass;

        $this->classes = array_fill_keys($classes, 0.0);

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
        [$samples, $labels, $distances] = $this->tree->range($sample, $this->radius);

        if (empty($labels)) {
            return $this->outlierClass;
        }

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
     * @param list<string|int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        [$samples, $labels, $distances] = $this->tree->range($sample, $this->radius);

        $dist = $this->classes;

        if (empty($labels)) {
            $dist[$this->outlierClass] = 1.0;

            return $dist;
        }

        if ($this->weighted) {
            $weights = array_fill_keys($labels, 0.0);

            foreach ($labels as $i => $label) {
                $weights[$label] += 1.0 / (1.0 + $distances[$i]);
            }
        } else {
            $weights = array_count_values($labels);
        }

        $total = array_sum($weights);

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
        return 'Radius Neighbors (' . Params::stringify($this->params()) . ')';
    }
}
