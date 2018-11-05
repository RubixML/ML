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
use InvalidArgumentException;
use RuntimeException;

/**
 * K-d Neighbors
 *
 * A fast K Nearest Neighbors algorithm that uses a K-d tree to divide the training
 * set into neighborhoods whose max size are constrained by the neighborhood
 * hyperparameter. K-d Neighbors does a binary search to locate the nearest
 * neighborhood and then searches only the points close to or within the neighborhood
 * for the nearest k to make a prediction. The main advantage K-d Neighbors has over
 * regular brute force KNN is that it is faster, however it cannot be partially
 * trained.
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
    public function __construct(int $k = 3, int $neighborhood = 20, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException("At least 1 neighbor is required"
                . " to make a prediction, $k given.");
        }

        if ($k > $neighborhood) {
            throw new InvalidArgumentException("K cannot be larger than the max"
                . " size of a neighborhood, $k given but $neighborhood allowed.");
        }

        $this->k = $k;

        parent::__construct($neighborhood, $kernel);
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
     * Make predictions from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trainied.');
        }
        
        $predictions = [];

        foreach ($dataset as $sample) {
            list($labels, $distances) = $this->neighbors($sample, $this->k);

            $counts = array_count_values($labels);

            $predictions[] = Argmax::compute($counts);
        }

        return $predictions;
    }

    /**
     * Estimate probabilities for each possible outcome.
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

        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trainied.');
        }

        $probabilities = array_fill(0, $dataset->numRows(),
            array_fill_keys($this->classes, 0.));

        foreach ($dataset as $i => $sample) {
            list($labels, $distances) = $this->neighbors($sample, $this->k);

            $n = count($labels);

            foreach (array_count_values($labels) as $class => $count) {
                $probabilities[$i][$class] = $count / $n;
            }
        }

        return $probabilities;
    }
}
