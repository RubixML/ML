<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Helpers\Verifier;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;
use Stringable;

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
class RadiusNeighbors implements Estimator, Learner, Probabilistic, Persistable, Stringable
{
    use PredictsSingle, ProbaSingle;

    /**
     * The radius within which points are considered neighbors.
     *
     * @var float
     */
    protected $radius;

    /**
     * Should we consider the distances of our nearest neighbors when making predictions?
     *
     * @var bool
     */
    protected $weighted;

    /**
     * The spatial tree used to run range searches.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

    /**
     * The class label for any samples that have 0 neighbors within the specified radius.
     *
     * @var string
     */
    protected $outlierClass;

    /**
     * The zero vector for the possible class outcomes.
     *
     * @var float[]|null
     */
    protected $classes;

    /**
     * The dimensionality of the training set.
     *
     * @var int|null
     */
    protected $featureCount;

    /**
     * @param float $radius
     * @param bool $weighted
     * @param string $outlierClass
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $radius = 1.0,
        bool $weighted = true,
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
            'outlier_class' => $this->outlierClass,
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
     * @return \Rubix\ML\Graph\Trees\Spatial
     */
    public function tree() : Spatial
    {
        return $this->tree;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        Verifier::check([
            DatasetIsNotEmpty::with($dataset),
            SamplesAreCompatibleWithEstimator::with($dataset, $this),
            LabelsAreCompatibleWithLearner::with($dataset, $this),
        ]);

        $classes = $dataset->possibleOutcomes();

        if (in_array($this->outlierClass, $classes)) {
            throw new RuntimeException('Training set cannot contain'
                . ' labels of the outlier class.');
        }

        $classes[] = $this->outlierClass;

        $this->classes = array_fill_keys($classes, 0.0);

        $this->featureCount = $dataset->numColumns();

        $this->tree->grow($dataset);
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return list<string>
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->tree->bare() or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            [$samples, $labels, $distances] = $this->tree->range($sample, $this->radius);

            if (empty($labels)) {
                $predictions[] = $this->outlierClass;

                continue 1;
            }

            if ($this->weighted) {
                $weights = array_fill_keys($labels, 0.0);

                foreach ($labels as $i => $label) {
                    $weights[$label] += 1.0 / (1.0 + $distances[$i]);
                }
            } else {
                $weights = array_count_values($labels);
            }

            $predictions[] = argmax($weights);
        }

        return $predictions;
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->tree->bare() or !$this->classes or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $probabilities = [];

        foreach ($dataset->samples() as $sample) {
            [$samples, $labels, $distances] = $this->tree->range($sample, $this->radius);

            $dist = $this->classes;

            if (empty($labels)) {
                $dist[$this->outlierClass] = 1.0;

                $probabilities[] = $dist;

                continue 1;
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

            $probabilities[] = $dist;
        }

        return $probabilities;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Radius Neighbors (' . Params::stringify($this->params()) . ')';
    }
}
