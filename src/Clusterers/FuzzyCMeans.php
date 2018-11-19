<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Traits\LoggerAware;
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
 * [2] A. Stetco et al. (2015). Fuzzy C-means++: Fuzzy C-means with effective
 * seeding initialization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class FuzzyCMeans implements Learner, Probabilistic, Verbose, Persistable
{
    use LoggerAware;
    
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
        if ($dataset->typeCount(DataFrame::CONTINUOUS) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('This estimator only works'
                . ' with continuous features.');
        }

        if ($this->logger) $this->logger->info('Learner initialized w/ '
            . Params::stringify([
                'c' => $this->c,
                'fuzz' => $this->fuzz,
                'kernel' => $this->kernel,
                'epochs' => $this->epochs,
                'min_change' => $this->minChange,
            ]));

        if ($this->logger) $this->logger->info("Initializing $this->c"
            . ' cluster centroids');

        $this->centroids = $this->initializeCentroids($dataset);

        $this->steps = $memberships = [];

        $rotated = $dataset->columns();
        $previous = INF;

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
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

            if ($this->logger) $this->logger->info("Epoch $epoch"
                . " complete, loss=$loss");

            if (abs($previous - $loss) < $this->minChange) {
                break 1;
            }

            $previous = $loss;
        }

        if ($this->logger) $this->logger->info('Training complete');
    }

    /**
     * Make predictions from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([Argmax::class, 'compute'], $this->proba($dataset));
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

        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return array_map([self::class, 'calculateMembership'], $dataset->samples());
    }

        /**
     * Initialize the cluster centroids using the k-means++ method.
     * 
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    protected function initializeCentroids(Dataset $dataset) : array
    {
        $n = $dataset->numRows();

        if ($n < $this->c) {
            throw new RuntimeException('The number of samples cannot be less'
                . ' than the number of target clusters.');
        }

        $weights = array_fill(0, $n, 1. / $n);

        $centroids = [];

        for ($i = 0; $i < $this->c; $i++) {
            $subset = $dataset->randomWeightedSubsetWithReplacement(1, $weights);

            $centroids[] = $subset->row(0);

            if ($i === $this->c) {
                break 1;
            }

            foreach ($dataset as $j => $sample) {
                $closest = INF;

                foreach ($centroids as $centroid) {
                    $distance = $this->kernel->compute($sample, $centroid);

                    if ($distance < $closest) {
                        $closest = $distance;
                    }
                }

                $weights[$j] = $closest ** 2;
            }

            $total = array_sum($weights) ?: self::EPSILON;

            foreach ($weights as &$weight) {
                $weight /= $total;
            }
        }

        return $centroids;
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

                $total += ($a / ($b ?: self::EPSILON)) ** $this->lambda;
            }

            $membership[$cluster] = 1. / ($total ?: self::EPSILON);
        }

        return $membership;
    }

    /**
     * Calculate the inter-cluster distance between each training sample and
     * each cluster centroid.
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
