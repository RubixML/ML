<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\argmax;
use function count;
use function is_nan;

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
    use AutotrackRevisions, LoggerAware;

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
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected $steps;

    /**
     * @param int $c
     * @param float $fuzz
     * @param int $epochs
     * @param float $minChange
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param \Rubix\ML\Clusterers\Seeders\Seeder|null $seeder
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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
            throw new InvalidArgumentException('C must be greater'
                . " than 0, $c given.");
        }

        if ($fuzz <= 1.0) {
            throw new InvalidArgumentException('Fuzz factor must be'
                . " greater than 1, $fuzz given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
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
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::clusterer();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'c' => $this->c,
            'fuzz' => $this->fuzz,
            'epochs' => $this->epochs,
            'min_change' => $this->minChange,
            'kernel' => $this->kernel,
            'seeder' => $this->seeder,
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
     * Return the loss at each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function steps() : ?array
    {
        return $this->steps;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        if ($this->logger) {
            $this->logger->info("$this initialized");
        }

        $this->centroids = $this->seeder->seed($dataset, $this->c);

        $this->steps = [];

        $columns = $dataset->columns();

        $prevLoss = INF;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $memberships = array_map([$this, 'probaSample'], $dataset->samples());

            $loss = $this->inertia($dataset->samples(), $memberships);

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->info('Numerical instability detected');
                }

                break;
            }

            $loss /= $dataset->numRows();

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - Inertia: $loss");
            }

            foreach ($this->centroids as $cluster => &$centroid) {
                $means = [];

                foreach ($columns as $column => $values) {
                    $sigma = $total = 0.0;

                    foreach ($memberships as $i => $probabilities) {
                        $weight = $probabilities[$cluster] ** $this->fuzz;

                        $sigma += $weight * $values[$i];
                        $total += $weight;
                    }

                    $means[] = $sigma / $total;
                }

                $centroid = $means;
            }

            if ($loss <= 0.0) {
                break;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break;
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
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->centroids)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param (int|float)[] $sample
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        return argmax($this->probaSample($sample));
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->centroids)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the c centroids.
     *
     * @param list<int|float> $sample
     * @return float[]
     */
    protected function probaSample(array $sample) : array
    {
        $distances = $dist = [];

        foreach ($this->centroids as $centroid) {
            $distances[] = $this->kernel->compute($sample, $centroid) ?: EPSILON;
        }

        foreach ($distances as $distanceA) {
            $sigma = 0.0;

            foreach ($distances as $distanceB) {
                $sigma += ($distanceA / $distanceB) ** $this->rho;
            }

            $dist[] = 1.0 / $sigma;
        }

        return $dist;
    }

    /**
     * Calculate the  sum of distances between all samples and their closest centroid.
     *
     * @param array[] $samples
     * @param array[] $memberships
     * @return float
     */
    protected function inertia(array $samples, array $memberships) : float
    {
        $inertia = 0.0;

        foreach ($samples as $i => $sample) {
            $membership = $memberships[$i];

            foreach ($this->centroids as $cluster => $centroid) {
                $inertia += $membership[$cluster] * $this->kernel->compute($sample, $centroid);
            }
        }

        return $inertia;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Fuzzy C Means (' . Params::stringify($this->params()) . ')';
    }
}
