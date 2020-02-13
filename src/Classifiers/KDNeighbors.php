<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Other\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\argmax;

use const Rubix\ML\EPSILON;

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
    use PredictsSingle, ProbaSingle;
    
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
     * The spatial tree used to run nearest neighbor searches.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

    /**
     * The zero vector for the possible class outcomes.
     *
     * @var float[]|null
     */
    protected $classes;

    /**
     * @param int $k
     * @param bool $weighted
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 5, bool $weighted = true, ?Spatial $tree = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->tree = $tree ?? new KDTree();
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->tree->kernel()->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
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
     * @var \Rubix\ML\Graph\Trees\Spatial
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
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $this->classes = array_fill_keys($dataset->possibleOutcomes(), 0.0);

        $this->tree->grow($dataset);
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->tree->bare()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            [$samples, $labels, $distances] = $this->tree->nearest($sample, $this->k);

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
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->tree->bare() or !$this->classes) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = [];

        foreach ($dataset->samples() as $sample) {
            [$samples, $labels, $distances] = $this->tree->nearest($sample, $this->k);

            if ($this->weighted) {
                $weights = array_fill_keys($labels, 0.0);

                foreach ($labels as $i => $label) {
                    $weights[$label] += 1.0 / (1.0 + $distances[$i]);
                }
            } else {
                $weights = array_count_values($labels);
            }

            $total = array_sum($weights) ?: EPSILON;

            $dist = $this->classes;

            foreach ($weights as $class => $weight) {
                $dist[$class] = $weight / $total;
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
    }
}
