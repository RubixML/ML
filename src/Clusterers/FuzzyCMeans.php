<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function count;

use const Rubix\ML\EPSILON;

/**
 * Fuzzy C Means
 *
 * A distance-based soft clustering algorithm that allows samples to belong to multiple
 * clusters if they fall within a *fuzzy* region controlled by the fuzz parameter. Like
 * K Means, Fuzzy C Means minimizes the inertia cost function, however, unlike K Means,
 * FCM uses a batch solver that requires the entire dataset to compute the update to the
 * cluster centroids at each iteration.
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
    use PredictsSingle, ProbaSingle, LoggerAware;
    
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
    protected $rho;

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
     * The distance kernel to use when computing the distances between samples.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The cluster centroid seeder.
     *
     * @var \Rubix\ML\Clusterers\Seeders\Seeder
     */
    protected $seeder;

    /**
     * The computed centroid vectors of the training data.
     *
     * @var array[]
     */
    protected $centroids = [
        //
    ];

    /**
     * The inertia at each epoch of training.
     *
     * @var float[]
     */
    protected $steps = [
        //
    ];

    /**
     * @param int $c
     * @param float $fuzz
     * @param int $epochs
     * @param float $minChange
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param \Rubix\ML\Clusterers\Seeders\Seeder|null $seeder
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $c,
        float $fuzz = 2.0,
        int $epochs = 300,
        float $minChange = 1e-4,
        ?Distance $kernel = null,
        ?Seeder $seeder = null
    ) {
        if ($c < 1) {
            throw new InvalidArgumentException('Must target at least one'
                . " cluster, $c given.");
        }

        if ($fuzz <= 1.0) {
            throw new InvalidArgumentException('Fuzz factor must be greater'
                . " than 1, $fuzz given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . " least 1 epoch, $epochs given.");
        }
        
        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . " than 0, $minChange given.");
        }

        $this->c = $c;
        $this->fuzz = $fuzz;
        $this->rho = 2.0 / ($fuzz - 1.0);
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->kernel = $kernel ?? new Euclidean();
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
     * @return array[]
     */
    public function centroids() : array
    {
        return $this->centroids;
    }

    /**
     * Return the inter cluster distance at each epoch of training.
     *
     * @return float[]
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
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner init ' . Params::stringify([
                'c' => $this->c,
                'fuzz' => $this->fuzz,
                'epochs' => $this->epochs,
                'min_change' => $this->minChange,
                'kernel' => $this->kernel,
                'seeder' => $this->seeder,
            ]));
        }

        $this->centroids = $this->seeder->seed($dataset, $this->c);

        $this->steps = [];

        $columns = $dataset->columns();

        $prevLoss = INF;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $memberships = array_map([self::class, 'membership'], $dataset->samples());

            foreach ($this->centroids as $cluster => &$centroid) {
                foreach ($columns as $column => $values) {
                    $sigma = $total = 0.0;

                    foreach ($memberships as $i => $probabilities) {
                        $weight = $probabilities[$cluster] ** $this->fuzz;

                        $sigma += $weight * $values[$i];
                        $total += $weight;
                    }

                    $centroid[$column] = $sigma / ($total ?: EPSILON);
                }
            }

            $loss = $this->inertia($dataset->samples(), $memberships);

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch loss=$loss");
            }

            if (is_nan($loss) or $loss < EPSILON) {
                break 1;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break 1;
            }

            $prevLoss = $loss;
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return int[]
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
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'membership'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the c centroids.
     *
     * @param (string|int|float)[] $sample
     * @return float[]
     */
    protected function membership(array $sample) : array
    {
        $membership = $deltas = [];

        foreach ($this->centroids as $centroid) {
            $deltas[] = $this->kernel->compute($sample, $centroid) ?: EPSILON;
        }

        foreach ($this->centroids as $cluster => $centroid) {
            $distance = $this->kernel->compute($sample, $centroid);

            $sigma = 0.0;

            foreach ($deltas as $delta) {
                $sigma += ($distance / $delta) ** $this->rho;
            }

            $membership[$cluster] = 1.0 / ($sigma ?: EPSILON);
        }

        return $membership;
    }

    /**
     * Calculate the average sum of distances between all samples and their closest
     * centroid.
     *
     * @param array[] $samples
     * @param array[] $memberships
     * @return float
     */
    protected function inertia(array $samples, array $memberships) : float
    {
        $n = count($samples);

        if ($n < 1) {
            return 0.0;
        }

        $inertia = 0.0;

        foreach ($samples as $i => $sample) {
            $membership = $memberships[$i];

            foreach ($this->centroids as $cluster => $centroid) {
                $inertia += $membership[$cluster]
                    * $this->kernel->compute($sample, $centroid);
            }
        }

        return $inertia / $n;
    }
}
