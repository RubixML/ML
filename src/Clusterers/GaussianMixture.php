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
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;
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
use function Rubix\ML\logsumexp;
use function Rubix\ML\array_transpose;
use function is_nan;

use const Rubix\ML\TWO_PI;
use const Rubix\ML\EPSILON;

/**
 * Gaussian Mixture
 *
 * A Gaussian Mixture model (GMM) is a probabilistic model for representing the presence
 * of clusters within an overall population without requiring a sample to know which
 * sub-population it belongs to beforehand. GMMs are similar to centroid-based clusterers
 * like [K Means](k-means.md) but allow both the cluster centers (*means*) as well as the
 * radii (*variances*) to be learned as well. For this reason, GMMs are especially useful
 * for clusterings that are of different radius.
 *
 * References:
 * [1] A. P. Dempster et al. (1977). Maximum Likelihood from Incomplete Data via
 * the EM Algorithm.
 * [2] J. Blomer et al. (2016). Simple Methods for Initializing the EM Algorithm
 * for Gaussian Mixture Models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianMixture implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The number of gaussian components to fit to the training set i.e. the
     * target number of clusters.
     *
     * @var int
     */
    protected $k;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The minimum shift in the components necessary to continue training.
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
     * The precomputed log prior probabilities of each cluster.
     *
     * @var float[]
     */
    protected $logPriors = [
        //
    ];

    /**
     * The computed means of each feature column for each gaussian.
     *
     * @var array[]
     */
    protected $means = [
        //
    ];

    /**
     * The computed variances of each feature column for each gaussian.
     *
     * @var array[]
     */
    protected $variances = [
        //
    ];

    /**
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected $steps;

    /**
     * @param int $k
     * @param int $epochs
     * @param float $minChange
     * @param \Rubix\ML\Clusterers\Seeders\Seeder|null $seeder
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        int $k,
        int $epochs = 100,
        float $minChange = 1e-3,
        ?Seeder $seeder = null
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be greater'
                . " than 0, $k given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        $this->k = $k;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->seeder = $seeder ?? new PlusPlus();
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
            'k' => $this->k,
            'epochs' => $this->epochs,
            'min_change' => $this->minChange,
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
        return $this->means and $this->variances;
    }

    /**
     * Return the cluster prior probabilities.
     *
     * @return float[]
     */
    public function priors() : array
    {
        return array_map('exp', $this->logPriors);
    }

    /**
     * Return the mean vectors of each component.
     *
     * @return array[]
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * Return the multivariate variance of each component.
     *
     * @return array[]
     */
    public function variances() : array
    {
        return $this->variances;
    }

    /**
     * Return the loss at each epoch of training from the last training session.
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

        $n = $dataset->numRows();

        $samples = $dataset->samples();
        $columns = $dataset->columns();

        $this->logPriors = array_fill(0, $this->k, log(1.0 / $this->k));

        [$this->means, $this->variances] = $this->initialize($dataset);

        $this->steps = [];

        $prevLoss = INF;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $memberships = [];
            $loss = 0.0;

            foreach ($samples as $sample) {
                $jll = $this->jointLogLikelihood($sample);

                $total = logsumexp($jll);

                $loss -= $total;

                $dist = [];

                foreach ($jll as $cluster => $likelihood) {
                    $dist[$cluster] = exp($likelihood - $total);
                }

                $memberships[] = $dist;
            }

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->info('Numerical instability detected');
                }

                break;
            }

            $loss /= $n;

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - loss: $loss");
            }

            for ($cluster = 0; $cluster < $this->k; ++$cluster) {
                $mHat = array_column($memberships, $cluster);

                $total = array_sum($mHat);

                $means = $variances = [];

                foreach ($columns as $column) {
                    $sigma = $ssd = 0.0;

                    foreach ($column as $i => $value) {
                        $sigma += $mHat[$i] * $value;
                    }

                    $mean = $sigma / $total;

                    foreach ($column as $i => $value) {
                        $ssd += $mHat[$i] * ($value - $mean) ** 2;
                    }

                    $variance = $ssd / $total;

                    $means[] = $mean;
                    $variances[] = $variance ?: EPSILON;
                }

                $logPrior = log($total / $n);

                $this->means[$cluster] = $means;
                $this->variances[$cluster] = $variances;
                $this->logPriors[$cluster] = $logPrior;
            }

            if ($loss <= 0.0) {
                break;
            }

            if (abs($loss - $prevLoss) < $this->minChange) {
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
        if (empty($this->logPriors)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->means) ?: []))->check();

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
        return argmax($this->jointLogLikelihood($sample));
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
        if (empty($this->logPriors)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->means) ?: []))->check();

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
        $jll = $this->jointLogLikelihood($sample);

        $total = logsumexp($jll);

        $dist = [];

        foreach ($jll as $cluster => $likelihood) {
            $dist[$cluster] = exp($likelihood - $total);
        }

        return $dist;
    }

    /**
     * Calculate the joint log likelihood of a sample being a member
     * of each of the gaussian components.
     *
     * @param list<int|float> $sample
     * @return float[]
     */
    protected function jointLogLikelihood(array $sample) : array
    {
        $likelihoods = [];

        foreach ($this->logPriors as $cluster => $prior) {
            $means = $this->means[$cluster];
            $variances = $this->variances[$cluster];

            $likelihood = $prior;

            foreach ($sample as $column => $feature) {
                $mean = $means[$column];
                $variance = $variances[$column];

                $pdf = -0.5 * log(TWO_PI * $variance);
                $pdf -= 0.5 * (($feature - $mean) ** 2) / $variance;

                $likelihood += $pdf;
            }

            $likelihoods[$cluster] = $likelihood;
        }

        return $likelihoods;
    }

    /**
     * Initialize the gaussian components by calculating the means and
     * variances of k initial cluster centroids generated by the seeder.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    protected function initialize(Dataset $dataset) : array
    {
        $kernel = new Euclidean();

        /** @var list<list<int|float>> $centroids */
        $centroids = $this->seeder->seed($dataset, $this->k);

        $clusters = array_fill(0, $this->k, []);

        foreach ($dataset->samples() as $sample) {
            $bestDistance = INF;
            $bestCluster = -1;

            foreach ($centroids as $cluster => $centroid) {
                $distance = $kernel->compute($sample, $centroid);

                if ($distance < $bestDistance) {
                    $bestDistance = $distance;
                    $bestCluster = $cluster;
                }
            }

            $clusters[$bestCluster][] = $sample;
        }

        $means = $variances = [];

        foreach ($clusters as $cluster => $samples) {
            $mHat = $vHat = [];

            $columns = array_transpose($samples);

            foreach ($columns as $values) {
                [$mean, $variance] = Stats::meanVar($values);

                $mHat[] = $mean;
                $vHat[] = $variance ?: EPSILON;
            }

            $means[$cluster] = $mHat;
            $variances[$cluster] = $vHat;
        }

        return [$means, $variances];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Gaussian Mixture (' . Params::stringify($this->params()) . ')';
    }
}
