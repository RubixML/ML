<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use RuntimeException;

/**
 * Fuzzy C Means
 *
 * Clusterer that allows data points to belong to multiple clusters if they fall
 * within a fuzzy region and thus is able to output probabilities for each
 * cluster via the Probabilistic interface.
 *
 * References:
 * [1] J. C. Bezdek et al. (1984). FCM: The Fuzzy C-Means Clustering Algorithm.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class FuzzyCMeans implements Learner, Probabilistic, Persistable
{
    /**
     * The target number of clusters.
     *
     * @var int
     */
    protected $c;

    /**
     * This determines the bandwidth of the fuzzy area. i.e. The fuzz factor.
     *
     * @var float
     */
    protected $fuzz;

    /**
     * The memoized exponent of the membership calculation.
     *
     * @var float
     */
    protected $lambda;

    /**
     * The distance kernel to use when computing the distances between
     * samples.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The minimum change in the centroids necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The computed centroid vectors of the training data.
     *
     * @var array
     */
    protected $centroids = [
        //
    ];

    /**
     * The inter cluster distances at each epoch of training.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param  int  $c
     * @param  float  $fuzz
     * @param  \Rubix\ML\Kernels\Distance\Distance|null  $kernel
     * @param  int  $epochs
     * @param  float  $minChange
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $c, float $fuzz = 2.0, ?Distance $kernel = null,
                                int $epochs = PHP_INT_MAX, float $minChange = 1e-4)
    {
        if ($c < 1) {
            throw new InvalidArgumentException("Must target at least one"
                . " cluster, $c given.");
        }

        if ($fuzz <= 1.) {
            throw new InvalidArgumentException("Fuzz factor must be greater"
                . " than 1, $fuzz given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException("Estimator must train for at"
                . " least 1 epoch, $epochs given.");
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException("Minimum change cannot be less"
                . " than 0, $minChange given.");
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->c = $c;
        $this->fuzz = $fuzz;
        $this->lambda = 2. / ($fuzz - 1.);
        $this->kernel = $kernel;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLUSTERER;
    }

    /**
     * Return the computed cluster centroids of the training data.
     *
     * @return array
     */
    public function centroids() : array
    {
        return $this->centroids;
    }

    /**
     * Return the inter cluster distance at each epoch of training.
     *
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Pick C random samples and assign them as centroids. Compute the coordinates
     * of the centroids by clustering the points based on each sample's distance
     * from one of the C centroids, then recompute the centroid coordinate as the
     * mean of the new cluster.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if ($dataset->numRows() < $this->c) {
            throw new RuntimeException('The number of samples cannot be less'
                . ' than the parameter C.');
        }

        $this->centroids = $dataset->randomize()->tail($this->c)->samples();

        $this->steps = $memberships = [];

        $rotated = $dataset->rotate();
        $previous = INF;

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            foreach ($dataset as $i => $sample) {
                $memberships[$i] = $this->calculateMembership($sample);
            }

            foreach ($this->centroids as $cluster => &$centroid) {
                foreach ($rotated as $column => $values) {
                    $sigma = $total = 0.;

                    foreach ($memberships as $i => $probabilities) {
                        $weight = $probabilities[$cluster] ** $this->fuzz;

                        $sigma += $weight * $values[$i];
                        $total += $weight;
                    }

                    $centroid[$column] = $sigma / ($total ?: self::EPSILON);
                }
            }

            $loss = $this->interClusterDistance($dataset, $memberships);

            $this->steps[] = $loss;

            if (abs($previous - $loss) < $this->minChange) {
                break 1;
            }

            $previous = $loss;
        }
    }

    /**
     * Make a prediction based on the cluster probabilities.
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
     * Return an array of cluster probabilities for each sample.
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

        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = [];

        foreach ($dataset as $sample) {
            $probabilities[] = $this->calculateMembership($sample);
        }

        return $probabilities;
    }

    /**
     * Return an vector of membership probability score of each cluster for a
     * given sample.
     *
     * @param  array  $sample
     * @return array
     */
    protected function calculateMembership(array $sample) : array
    {
        $membership = [];

        foreach ($this->centroids as $cluster => $centroid1) {
            $a = $this->kernel->compute($sample, $centroid1);

            $total = 0.;

            foreach ($this->centroids as $centroid2) {
                $b = $this->kernel->compute($sample, $centroid2);

                $total += ($b !== 0. ? ($a / $b) : 1.) ** $this->lambda;
            }

            $membership[$cluster] = 1. / $total;
        }

        return $membership;
    }

    /**
     * Calculate the inter-cluster distance.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return float
     */
    protected function interClusterDistance(Dataset $dataset, array $memberships) : float
    {
        $distance = 0.;

        foreach ($dataset as $i => $sample) {
            $membership = $memberships[$i];

            foreach ($this->centroids as $cluster => $centroid) {
                $distance += $membership[$cluster]
                    * $this->kernel->compute($sample, $centroid);
            }
        }

        return $distance;
    }
}
