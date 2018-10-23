<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use RuntimeException;

/**
 * K-d Neighbors Regressor
 *
 * A fast approximating implementation of KNN Regressor using a K-d tree. The KDN
 * Regressor works by locating the neighborhood of a sample via binary search and
 * then does a brute force search only on the samples in the neighborhood. The
 * main advantage of K-d Neighbors over KNN is speed and added variance to the
 * predictions (if that is desired).
 *
 * [1] J. L. Bentley. (1975). Multidimensional Binary Seach Trees Used for
 * Associative Searching.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDNRegressor extends KDTree implements Learner, Persistable
{
    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected $k;

    /**
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * @param  int  $k
     * @param  int  $neighborhood
     * @param  \Rubix\ML\Kernels\Distance\Distance|null  $kernel
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 3, int $neighborhood = 10, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException("At least 1 neighbor is required"
                . " to make a prediction, $k given.");
        }

        if ($k > $neighborhood) {
            throw new InvalidArgumentException("K cannot be larger than the"
                . " size of the neighborhood, $k given but $neighborhood"
                . " allowed.");
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->k = $k;
        $this->kernel = $kernel;

        parent::__construct($neighborhood);
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->grow($dataset);
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if ($this->bare() === true) {
            throw new RuntimeException('Estimator has not been trainied.');
        }

        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = Stats::mean($this->findNearestNeighbors($sample));
        }

        return $predictions;
    }

    /**
     * Find the k nearest neighbors to the given sample from a neighborhood.
     *
     * @param  array  $sample
     * @return array
     */
    public function findNearestNeighbors(array $sample) : array
    {
        $neighborhood = $this->search($sample);

        $distances = [];

        foreach ($neighborhood->samples() as $i => $neighbor) {
            $distances[$i] = $this->kernel->compute($sample, $neighbor);
        }

        asort($distances);

        return array_intersect_key($neighborhood->labels(),
            array_slice($distances, 0, $this->k, true));

    }
}
