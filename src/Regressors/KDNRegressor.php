<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;

/**
 * K-d Neighbors Regressor
 *
 * A fast implementation of K Nearest Neighbors regression using a K-d tree. The
 * advantage of K-d Neighbors over KNN is speed and added variance to the
 * predictions (if that is desired).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDNRegressor extends KDTree implements Regressor, Persistable
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

        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->grow($dataset);
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = Average::mean($this->findNearestNeighbors($sample));
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

        $labels = $neighborhood->labels();

        $distances = [];

        foreach ($neighborhood->samples() as $i => $neighbor) {
            $distances[$i] = $this->kernel->compute($sample, $neighbor);
        }

        asort($distances);

        return array_intersect_key($labels,
            array_slice($distances, 0, $this->k, true));

    }
}
