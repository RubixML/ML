<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\KDTree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\Distance;
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
class KDNeighborsRegressor extends KDTree implements Learner, Persistable
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
     * @param int $k
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param bool $weighted
     * @param int $maxLeafSize
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $k = 3,
        ?Distance $kernel = null,
        bool $weighted = true,
        int $maxLeafSize = 30
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
        return !$this->bare();
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->grow($dataset);
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
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trainied.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            [$labels, $distances] = $this->nearest($sample, $this->k);

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
