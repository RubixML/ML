<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Functions\ArgMax;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;

/**
 * K Nearest Neighbors
 *
 * A distance-based algorithm that locates the K nearest neighbors from the
 * training set and uses a majority vote to classify the unknown sample. K
 * Nearest Neighbors is considered a lazy learning Estimator because it does all
 * of its computation at prediction time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KNearestNeighbors implements Multiclass, Online, Probabilistic, Persistable
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
     * The training samples that make up the neighborhood of the problem space.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The memoized labels of the training set.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * @param  int  $k
     * @param  \Rubix\ML\Kernels\Distance\Distance  $kernel
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 5, Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . ' to make a prediction.');
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->k = $k;
        $this->kernel = $kernel;
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

        $this->classes = $this->samples = $this->labels = [];

        $this->partial($dataset);
    }

    /**
     * Store the sample and outcome arrays. No other work to be done as this is
     * a lazy learning algorithm.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->classes = array_merge($this->classes, $dataset->possibleOutcomes());
        $this->samples = array_merge($this->samples, $dataset->samples());
        $this->labels = array_merge($this->labels, $dataset->labels());
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

        foreach ($this->proba($dataset) as $probabilities) {
            $predictions[] = ArgMax::compute($probabilities);
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $probabilities = array_fill(0, $dataset->numRows(),
            array_fill_keys($this->classes, 0.0));

        foreach ($dataset as $i => $sample) {
            $neighbors = $this->findNearestNeighbors($sample);

            $n = count($neighbors);

            foreach (array_count_values($neighbors) as $class => $count) {
                $probabilities[$i][$class] = $count / ($n + self::EPSILON);
            }
        }

        return $probabilities;
    }

    /**
     * Find the K nearest neighbors to the given sample vector.
     *
     * @param  array  $sample
     * @return array
     */
    protected function findNearestNeighbors(array $sample) : array
    {
        $distances = [];

        foreach ($this->samples as $index => $neighbor) {
            $distances[$index] = $this->kernel->compute($sample, $neighbor);
        }

        asort($distances);

        return array_intersect_key($this->labels,
            array_slice($distances, 0, $this->k, true));
    }
}
