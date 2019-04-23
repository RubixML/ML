<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Fuzzy C Means
 *
 * Distance-based soft clusterer that allows samples to belong to multiple clusters
 * if they fall within a *fuzzy* region controlled by the *fuzz* parameter.
 *
 * References:
 * [1] J. C. Bezdek et al. (1984). FCM: The Fuzzy C-Means Clustering Algorithm.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class FuzzyCMeans implements Estimator, Learner, Probabilistic, Verbose, Persistable
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
     * The precomputed exponent of the membership calculation.
     *
     * @var float
     */
    protected $lambda;

    /**
     * The distance kernel to use when computing the distances between samples.
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
     * The minimum change in inertia to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The cluster centroid seeder.
     *
     * @var \Rubix\ML\Clusterers\Seeders\Seeder
     */
    protected $seeder;

    /**
     * The computed centroid vectors of the training data.
     *
     * @var array
     */
    protected $centroids = [
        //
    ];

    /**
     * The inertia at each epoch of training.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param int $c
     * @param float $fuzz
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param int $epochs
     * @param float $minChange
     * @param \Rubix\ML\Clusterers\Seeders\Seeder|null $seeder
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $c,
        float $fuzz = 2.0,
        ?Distance $kernel = null,
        int $epochs = 300,
        float $minChange = 10.,
        ?Seeder $seeder = null
    ) {
        if ($c < 1) {
            throw new InvalidArgumentException('Must target at least one'
                . " cluster, $c given.");
        }

        if ($fuzz <= 1.) {
            throw new InvalidArgumentException('Fuzz factor must be greater'
                . " than 1, $fuzz given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . " least 1 epoch, $epochs given.");
        }
        
        if ($minChange < 0.) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . " than 0, $minChange given.");
        }

        $this->c = $c;
        $this->fuzz = $fuzz;
        $this->lambda = 2. / ($fuzz - 1.);
        $this->kernel = $kernel ?? new Euclidean();
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->seeder = $seeder ?? new PlusPlus($kernel);
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
        return [
            DataType::CONTINUOUS,
        ];
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
     * Return the inter cluster distance at each epoch of training.
     *
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner initialized w/ '
                . Params::stringify([
                    'c' => $this->c,
                    'fuzz' => $this->fuzz,
                    'kernel' => $this->kernel,
                    'epochs' => $this->epochs,
                    'min_change' => $this->minChange,
                    'seeder' => $this->seeder,
                ]));
        }

        $this->centroids = $this->seeder->seed($dataset, $this->c);

        $this->steps = $memberships = [];

        $samples = $dataset->samples();
        $rotated = $dataset->columns();

        $previous = INF;

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $memberships = array_map([self::class, 'membership'], $samples);

            foreach ($this->centroids as $cluster => &$centroid) {
                foreach ($rotated as $column => $values) {
                    $sigma = $total = 0.;

                    foreach ($memberships as $i => $probabilities) {
                        $weight = $probabilities[$cluster] ** $this->fuzz;

                        $sigma += $weight * $values[$i];
                        $total += $weight;
                    }

                    $centroid[$column] = $sigma / ($total ?: EPSILON);
                }
            }

            $inertia = $this->inertia($dataset, $memberships);

            $this->steps[] = $inertia;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch complete, inertia=$inertia");
            }

            if (is_nan($inertia)) {
                break 1;
            }

            if (abs($previous - $inertia) < $this->minChange) {
                break 1;
            }

            $previous = $inertia;
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map('Rubix\ML\argmax', $this->proba($dataset));
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'membership'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the c centroids.
     *
     * @param array $sample
     * @return array
     */
    protected function membership(array $sample) : array
    {
        $membership = $deltas = [];

        foreach ($this->centroids as $centroid) {
            $deltas[] = $this->kernel->compute($sample, $centroid);
        }

        foreach ($this->centroids as $cluster => $centroid) {
            $alpha = $this->kernel->compute($sample, $centroid);

            $sigma = 0.;

            foreach ($deltas as $delta) {
                $sigma += ($alpha / ($delta ?: EPSILON)) ** $this->lambda;
            }

            $membership[$cluster] = 1. / ($sigma ?: EPSILON);
        }

        return $membership;
    }

    /**
     * Calculate the sum of distances between all samples and their closest
     * centroid.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param array $memberships
     * @return float
     */
    protected function inertia(Dataset $dataset, array $memberships) : float
    {
        $inertia = 0.;

        foreach ($dataset as $i => $sample) {
            $membership = $memberships[$i];

            foreach ($this->centroids as $cluster => $centroid) {
                $inertia += $membership[$cluster]
                    * $this->kernel->compute($sample, $centroid);
            }
        }

        return $inertia;
    }
}
