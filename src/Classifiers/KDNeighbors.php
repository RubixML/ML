<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use RuntimeException;

/**
 * K-d Neighbors
 *
 * A fast K Nearest Neighbors approximating algorithm that uses a K-d tree to
 * divide the training set into neighborhoods whose max size are constrained by
 * the neighborhood hyperparameter. K-d Neighbors does a binary search to locate
 * the nearest neighborhood and then searches only the points in the neighborhood
 * for the nearest k to make a prediction. Since there may be points in other
 * neighborhoods that may be closer, the KNN search is said to be approximate.
 * The main advantage K-d Neighbors has over regular KNN is that it is much
 * faster.
 *
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
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The unique class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

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
        return self::CLASSIFIER;
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

        $this->classes = $dataset->possibleOutcomes();

        $this->grow($dataset);
    }

    /**
     * Make a prediction based on the class probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($this->proba($dataset) as $joint) {
            $predictions[] = Argmax::compute($joint);
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if ($this->bare() === true) {
            throw new RuntimeException('Estimator has not been trainied.');
        }

        $probabilities = array_fill(0, $dataset->numRows(),
            array_fill_keys($this->classes, 0.));

        foreach ($dataset as $i => $sample) {
            $neighbors = $this->findNearestNeighbors($sample);

            $n = count($neighbors);

            foreach (array_count_values($neighbors) as $class => $count) {
                $probabilities[$i][$class] = $count / $n;
            }
        }

        return $probabilities;
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
