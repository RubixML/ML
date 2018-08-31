<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use RuntimeException;

/**
 * K-d Neighbors Regressor
 *
 * A fast implementation of K Nearest Neighbors regression using a K-d tree. The
 * advantage of K-d Neighbors over KNN is speed and added variance to the
 * predictions (if that is desired).
 *
 * [1] J. L. Bentley. (1975). Multidimensional Binary Seach Trees Used for
 * Associative Searching.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDNRegressor extends KDTree implements Estimator, Persistable
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
     * @param  \Rubix\ML\Kernels\Distance\Distance  $kernel
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 3, int $neighborhood = 10, Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . ' to make a prediction.');
        }

        if ($k > $neighborhood) {
            throw new InvalidArgumentException('K cannot be larger than'
                . ' neighborhood.');
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
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        if (in_array(Dataset::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->grow($dataset);
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
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
