<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
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
     * This measures the tolerance the clusterer has to overlap and will
     * determine the bandwidth of the fuzzy area. i.e. The fuzz factor.
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
     * An array holding the progress of the last training session. i.e. the
     * total distance between all the centroids and data points.
     *
     * @var array
     */
    protected $progress = [
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
    public function __construct(int $c, float $fuzz = 2.0, Distance $kernel = null,
                            float $minChange = 1e-4, int $epochs = PHP_INT_MAX)
    {
        if ($c < 1) {
            throw new InvalidArgumentException('Target clusters must be'
                . ' greater than 1.');
        }

        if ($fuzz < 1) {
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

        if (!isset($kernel)) {
            $kernel = new Euclidean();
        }

        $this->c = $c;
        $this->fuzz = $fuzz + self::EPSILON;
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
     * Return the progress from last training session.
     *
     * @return array
     */
    public function progress() : array
    {
        return $this->progress;
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

        $this->centroids = array_fill(0, $this->c, array_fill(0,
            $dataset->numColumns(), 0.0));

        $memberships = $this->initializeMemberships($dataset->numRows());

        $this->step($dataset, $memberships);

        $this->progress = [];

        $previous = 0.0;

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($dataset as $index => $sample) {
                $memberships[$index] = $this->calculateMembership($sample);
            }

            $this->step($dataset, $memberships);

            $similarity = $this->calculateSimilarity($dataset);

            $this->progress[] = ['similarity' => $similarity];

            if (abs($similarity - $previous) < $this->minChange) {
                break 1;
            }

            $previous = $similarity;
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
            $best = ['probability' => -INF, 'outcome' => null];

            foreach ($probabilities as $label => $probability) {
                if ($probability > $best['probability']) {
                    $best['probability'] = $probability;
                    $best['outcome'] = $label;
                }
            }

            $predictions[] = $best['outcome'];
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

            $total = 0.0;

            foreach ($this->centroids as $centroid2) {
                $b = $this->kernel->compute($sample, $centroid2);

                $total += ($a / ($b + self::EPSILON))
                    ** (2 / ($this->fuzz - 1));
            }

            $membership[$label] = 1 / ($total + self::EPSILON);
        }

        return $membership;
    }

    /**
     * Update the cluster centroids with a new membership matrix.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @param  array  $memberships
     * @return void
     */
    protected function step(Dataset $dataset, array $memberships) : void
    {
        foreach ($this->centroids as $label => &$centroid) {
            foreach ($centroid as $j => &$mean) {
                $a = $total = self::EPSILON;

                foreach ($dataset as $k => $sample) {
                    $weight = $memberships[$k][$label] ** $this->fuzz;

                    $a += $weight * $sample[$j];
                    $total += $weight;
                }

                $mean = $a / $total;
            }
        }
    }

    /**
     * Initialize the membership matrix of dimension n x c.
     *
     * @param  int  $n
     * @return array
     */
    protected function initializeMemberships(int $n) : array
    {
        $memberships = array_fill(0, $n, array_fill(0, $this->c, 0.0));

        for ($i = 0; $i < $n; $i++) {
            $total = 0.0;

            for ($j = 0; $j < $this->c; $j++) {
                $weight = rand(0, (int) 1e8) / 1e8;

                $memberships[$i][$j] = $weight;

                $total += $weight;
            }

            foreach ($memberships[$i] as &$membership) {
                $membership /= $total;
            }
        }

        return $memberships;
    }

    /**
     * Return a similarity score inferred by maximizing the inter-cluster
     * distance.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return float
     */
    protected function calculateSimilarity(Dataset $dataset) : float
    {
        $similarity = 0.0;

        foreach ($dataset as $sample) {
            foreach ($this->centroids as $centroid) {
                $similarity += $this->kernel->compute($sample, $centroid);
            }
        }

        return $similarity;
    }
}
