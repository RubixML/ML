<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\argmax;

use const Rubix\ML\EPSILON;

/**
 * K-d Neighbors
 *
 * A fast K Nearest Neighbors algorithm that uses a K-d tree to divide the
 * training set into neighborhoods whose max size are controlled by the max
 * leaf size parameter. K-d Neighbors does a binary search to locate the
 * nearest neighborhood and then prunes all neighborhoods whose bounding box
 * is further than the kth nearest neighbor found so far. The main advantage
 * K-d Neighbors has over regular brute force KNN is that it is faster, however
 * it cannot be partially trained.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDNeighbors implements Estimator, Learner, Probabilistic, Persistable
{
    use PredictsSingle;
    
    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected $k;

    /**
     * The spatial tree used for nearest neighbor queries.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

    /**
     * Should we use the inverse distances as confidence scores when
     * making predictions?
     *
     * @var bool
     */
    protected $weighted;

    /**
     * The unique class labels.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param int $k
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @param bool $weighted
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 5, ?Spatial $tree = null, bool $weighted = true)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->tree = $tree ?? new KDTree();
        $this->weighted = $weighted;
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
        return [
            DataType::CONTINUOUS,
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
            throw new InvalidArgumentException('Estimator requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->classes = $dataset->possibleOutcomes();

        $this->tree->grow($dataset);
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
        if ($this->tree->bare()) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            [$samples, $labels, $distances] = $this->tree->nearest($sample, $this->k);

            if ($this->weighted) {
                $weights = array_fill_keys($labels, 0.);

                foreach ($labels as $i => $label) {
                    $weights[$label] += 1. / (1. + $distances[$i]);
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
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->tree->bare()) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $template = array_fill_keys($this->classes, 0.);

        $probabilities = [];

        foreach ($dataset as $sample) {
            [$samples, $labels, $distances] = $this->tree->nearest($sample, $this->k);

            if ($this->weighted) {
                $weights = array_fill_keys($labels, 0.);

                foreach ($labels as $i => $label) {
                    $weights[$label] += 1. / (1. + $distances[$i]);
                }
            } else {
                $weights = array_count_values($labels);
            }

            $total = array_sum($weights) ?: EPSILON;

            $dist = $template;

            foreach ($weights as $class => $weight) {
                $dist[$class] = $weight / $total;
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
    }
}
