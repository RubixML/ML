<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * K-d Neighbors Regressor
 *
 * A fast implementation of KNN Regressor using a K-d tree. The KDN Regressor works
 * by locating the neighborhood of a sample via binary search and then does a brute
 * force search only on the samples close to or within the neighborhood. The main
 * advantage of K-d Neighbors over brute force KNN is speed, however you no longer
 * have the ability to partially train.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDNeighborsRegressor implements Estimator, Learner, Persistable
{
    use PredictsSingle;
    
    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected $k;
    
    /**
     * Should we use the inverse distances as confidence scores when making
     * predictions?
     *
     * @var bool
     */
    protected $weighted;

    /**
     * The spatial tree used for nearest neighbor queries.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

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
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
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
     * Return the base k-d tree instance.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    public function tree() : Spatial
    {
        return $this->tree;
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->tree->grow($dataset);
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->tree->bare()) {
            throw new RuntimeException('Estimator has not been trainied.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            [$samples, $labels, $distances] = $this->tree->nearest($sample, $this->k);

            if ($this->weighted) {
                $weights = [];

                foreach ($distances as $distance) {
                    $weights[] = 1. / (1. + $distance);
                }

                $predictions[] = Stats::weightedMean($labels, $weights);
            } else {
                $predictions[] = Stats::mean($labels);
            }
        }

        return $predictions;
    }
}
