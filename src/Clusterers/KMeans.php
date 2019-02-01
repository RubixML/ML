<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * K Means
 *
 * A fast centroid-based hard clustering algorithm capable of clustering
 * linearly separable data points given a number of target clusters set by the
 * parameter K.
 *
 * References:
 * [1] T. Kanungo et al. (2002). An Efficient K-Means Clustering Algorithm:
 * Analysis and Implementation.
 * [2] D. Arthur et al. (2006). k-means++: The Advantages of Careful Seeding.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KMeans implements Online, Persistable
{
    /**
     * The target number of clusters.
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
     * @param  int  $k
     * @param  \Rubix\ML\Kernels\Distance\Distance|null  $kernel
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k, ?Distance $kernel = null, int $epochs = 300)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Must target at least one'
                . " cluster, $k given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . " least 1 epoch, $epochs given.");
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->k = $k;
        $this->kernel = $kernel;
        $this->epochs = $epochs;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLUSTERER;
    }

    /**
     * Return the data types that this estimator is compatible with.
     * 
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->kernel->compatibility();
    }

    /**
     * Has the learner been trained?
     * 
     * @return bool
     */
    public function trained() : bool
    {
        return !empty($this->centroids);
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
     * Train the learner with a dataset.
     * 
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->centroids = $this->initializeCentroids($dataset);

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function partial(Dataset $dataset) : void
    {
        if (empty($this->centroids)) {
            $this->train($dataset);

            return;
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $labels = array_fill(0, $dataset->numRows(), -1);
        $sizes = array_fill(0, $this->k, 0);

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            $changed = false;

            foreach ($dataset as $i => $sample) {
                $label = $this->assignCluster($sample);

                if ($label !== $labels[$i]) {
                    $labels[$i] = $label;

                    $sizes[$label]++;

                    $changed = true;
                }

                $size = $sizes[$label] ?: self::EPSILON;

                foreach ($this->centroids[$label] as $column => &$mean) {
                    $mean = ($mean * ($size - 1) + $sample[$column]) / $size;
                }
            }

            if (!$changed) {
                break 1;
            }
        }
    }

    /**
     * Cluster the dataset by assigning a label to each sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'assignCluster'], $dataset->samples());
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

        if ($n < $this->k) {
            throw new RuntimeException('The number of samples cannot be less'
                . ' than the number of target clusters.');
        }

        $weights = array_fill(0, $n, 1. / $n);

        $centroids = [];

        for ($i = 0; $i < $this->k; $i++) {
            $subset = $dataset->randomWeightedSubsetWithReplacement(1, $weights);

            $centroids[] = $subset[0];

            if ($i === $this->k) {
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
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param  array  $sample
     * @throws \RuntimeException
     * @return int
     */
    protected function assignCluster(array $sample) : int
    {
        $bestDistance = INF;
        $bestCluster = -1;

        foreach ($this->centroids as $cluster => $centroid) {
            $distance = $this->kernel->compute($sample, $centroid);

            if ($distance < $bestDistance) {
                $bestDistance = $distance;
                $bestCluster = $cluster;
            }
        }

        return (int) $bestCluster;
    }
}
