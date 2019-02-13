<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\KDTree;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * K-d Neighbors
 *
 * A fast K Nearest Neighbors algorithm that uses a K-d tree to divide the
 * training set into neighborhoods whose max size are controlled by the max
 * leaf size parameter. K-d Neighbors does a binary search to locate the
 * nearest neighborhood and then prunes all neighborhood whose bounding box
 * is further than the kth nearest neighbor found so far. The main advantage
 * K-d Neighbors has over regular brute force KNN is that it is faster, however
 * it cannot be partially trained.
 *
 * References:
 * [1] J. L. Bentley. (1975). Multidimensional Binary Seach Trees Used for
 * Associative Searching.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDNeighbors extends KDTree implements Learner, Probabilistic, Persistable
{
    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected $k;

    /**
     * Should we use the inverse distances as confidence scores when
     * making predictions?
     *
     * @var bool
     */
    protected $weighted;

    /**
     * The unique class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param int $k
     * @param int $maxLeafSize
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param bool $weighted
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $k = 3,
        int $maxLeafSize = 20,
        ?Distance $kernel = null,
        bool $weighted = true
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to make a prediction, $k given.");
        }

        if ($k > $maxLeafSize) {
            throw new InvalidArgumentException('K cannot be larger than the max'
                . " leaf size, $k given but $maxLeafSize allowed.");
        }

        $this->k = $k;
        $this->weighted = $weighted;

        parent::__construct($maxLeafSize, $kernel);
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->kernel->compatibility();
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->bare();
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
            throw new InvalidArgumentException('Estimator requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->classes = $dataset->possibleOutcomes();

        $this->grow($dataset);
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('The learner has not'
                . ' not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            [$labels, $distances] = $this->neighbors($sample);

            if ($this->weighted) {
                $weights = array_fill_keys($labels, 0.);

                foreach ($labels as $i => $label) {
                    $weights[$label] += 1. / (1. + $distances[$i]);
                }
            } else {
                $weights = array_count_values($labels);
            }

            $predictions[] = Argmax::compute($weights);
        }

        return $predictions;
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('The learner has not'
                . ' not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $probabilities = [];

        foreach ($dataset as $sample) {
            [$labels, $distances] = $this->neighbors($sample);

            if ($this->weighted) {
                $weights = array_fill_keys($labels, 0.);

                foreach ($labels as $i => $label) {
                    $weights[$label] += 1. / (1. + $distances[$i]);
                }
            } else {
                $weights = array_count_values($labels);
            }

            $total = array_sum($weights) ?: self::EPSILON;

            $dist = array_fill_keys($this->classes, 0.);

            foreach ($weights as $class => $weight) {
                $dist[$class] = $weight / $total;
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
    }
}
