<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
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
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class FuzzyCMeans implements Clusterer, Probabilistic, Persistable
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
    protected $distances = [
        //
    ];

    /**
     * @param  int  $c
     * @param  float  $fuzz
     * @param  \Rubix\ML\Kernels\Distance\Distance  $kernel
     * @param  float  $minChange
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $c, float $fuzz = 2.0, Distance $kernel = null, float $minChange = 1e-4,
                                int $epochs = PHP_INT_MAX)
    {
        if ($c < 1) {
            throw new InvalidArgumentException('Must target at least one'
                . ' cluster.');
        }

        if ($fuzz <= 1) {
            throw new InvalidArgumentException('Fuzz factor must be greater'
                . ' than 1.');
        }

        if ($minChange < 0) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . ' than 0.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->c = $c;
        $this->fuzz = $fuzz;
        $this->kernel = $kernel;
        $this->minChange = $minChange;
        $this->epochs = $epochs;
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
    public function distances() : array
    {
        return $this->distances;
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
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if ($dataset->numRows() < $this->c) {
            throw new RuntimeException('The number of samples cannot be less'
                . ' than the parameter C.');
        }

        $this->centroids = $dataset->randomize()->tail($this->c)->samples();

        $this->distances = $memberships = [];

        $previous = 0.0;

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            foreach ($dataset as $index => $sample) {
                $memberships[$index] = $this->calculateMembership($sample);
            }

            foreach ($this->centroids as $label => &$centroid) {
                foreach ($centroid as $column => &$mean) {
                    $sigma = $total = self::EPSILON;

                    foreach ($dataset as $index => $sample) {
                        $weight = $memberships[$index][$label] ** $this->fuzz;

                        $sigma += $weight * $sample[$column];
                        $total += $weight;
                    }

                    $mean = $sigma / $total;
                }
            }

            $distance = $this->computeInterClusterDistance($dataset, $memberships);

            $this->distances[] = $distance;

            if (abs($distance - $previous) < $this->minChange) {
                break 1;
            }

            $previous = $distance;
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

        foreach ($this->proba($dataset) as $probabilities) {
            $predictions[] = Argmax::compute($probabilities);
        }

        return $predictions;
    }

    /**
     * Return an array of cluster probabilities for each sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
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

        foreach ($this->centroids as $label => $centroid1) {
            $a = $this->kernel->compute($sample, $centroid1);

            $total = self::EPSILON;

            foreach ($this->centroids as $centroid2) {
                $b = $this->kernel->compute($sample, $centroid2);

                $total += ($a / ($b + self::EPSILON))
                    ** (2 / ($this->fuzz - 1));
            }

            $membership[$label] = 1 / $total;
        }

        return $membership;
    }

    /**
     * Return the inter-cluster distance.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return float
     */
    protected function computeInterClusterDistance(Dataset $dataset, array $memberships) : float
    {
        $distance = 0.0;

        foreach ($dataset as $i => $sample) {
            foreach ($this->centroids as $j => $centroid) {
                $distance += $memberships[$i][$j]
                    * $this->kernel->compute($sample, $centroid);
            }
        }

        return $distance;
    }
}
