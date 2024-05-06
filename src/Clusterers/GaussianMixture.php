<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\CPU;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function Rubix\ML\argmax;
use function Rubix\ML\logsumexp;
use function Rubix\ML\array_transpose;
use function array_column;
use function array_sum;
use function is_nan;
use function max;
use function abs;
use function log;
use function exp;
use function get_object_vars;

use const Rubix\ML\TWO_PI;

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
     * The number of gaussian components to fit to the training set i.e. the target number of clusters.
     *
     * @var int<0,max>
     */
    protected int $k;

    /**
     * The amount of epsilon smoothing added to the variance of each feature.
     *
     * @var float
     */
    protected float $smoothing;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * The minimum shift in the components necessary to continue training.
     *
     * @var float
     */
    protected float $minChange;

    /**
     * The cluster centroid seeder.
     *
     * @var Seeder
     */
    protected Seeder $seeder;

    /**
     * The precomputed log prior probabilities of each cluster.
     *
     * @var float[]
     */
    protected array $logPriors = [
        //
    ];

    /**
     * The computed means of each feature column for each gaussian.
     *
     * @var list<list<float>>
     */
    protected array $means = [
        //
    ];

    /**
     * The computed variances of each feature column for each gaussian.
     *
     * @var list<list<float>>
     */
    protected array $variances = [
        //
    ];

    /**
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param int $k
     * @param float $smoothing
     * @param int $epochs
     * @param float $minChange
     * @param Seeder|null $seeder
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $k,
        float $smoothing = 1e-9,
        int $epochs = 100,
        float $minChange = 1e-3,
        ?Seeder $seeder = null
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be greater'
                . " than 0, $k given.");
        }

        if ($smoothing <= 0.0) {
            throw new InvalidArgumentException('Smoothing must be'
                . " greater than 0, $smoothing given.");
        }

        if ($epochs < 0) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        $this->k = $k;
        $this->smoothing = $smoothing;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->seeder = $seeder ?? new PlusPlus();
    }

    /**
     * Return the estimator type.
     *
     * @return EstimatorType
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
            'smoothing' => $this->smoothing,
            'epochs' => $this->epochs,
            'min change' => $this->minChange,
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
     * @return list<list<float>>
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * Return the multivariate variance of each component.
     *
     * @return list<list<float>>
     */
    public function variances() : array
    {
        return $this->variances;
    }

    /**
     * Return an iterable progress table with the steps from the last training session.
     *
     * @return \Generator<mixed[]>
     */
    public function steps() : Generator
    {
        if (!$this->losses) {
            return;
        }

        foreach ($this->losses as $epoch => $loss) {
            yield [
                'epoch' => $epoch,
                'loss' => $loss,
            ];
        }
    }

    /**
     * Return the loss at each epoch of training from the last training session.
     *
     * @return float[]|null
     */
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        $this->initialize($dataset);

        $samples = $dataset->samples();
        $features = $dataset->features();

        $n = $dataset->numSamples();

        $minEpsilon = CPU::epsilon();
        $prevLoss = INF;

        $this->losses = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $loss = $maxVariance = 0.0;
            $memberships = [];

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
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            for ($cluster = 0; $cluster < $this->k; ++$cluster) {
                $affinities = array_column($memberships, $cluster);

                $total = array_sum($affinities);

                $means = $variances = [];

                foreach ($features as $column) {
                    $sigma = $ssd = 0.0;

                    foreach ($column as $i => $value) {
                        $sigma += $affinities[$i] * $value;
                    }

                    $mean = $sigma / $total;

                    foreach ($column as $i => $value) {
                        $ssd += $affinities[$i] * ($value - $mean) ** 2;
                    }

                    $variance = $ssd / $total;

                    $means[] = $mean;
                    $variances[] = $variance;
                }

                $maxVariance = max($maxVariance, ...$variances);

                $logPrior = log($total / $n);

                $this->means[$cluster] = $means;
                $this->variances[$cluster] = $variances;
                $this->logPriors[$cluster] = $logPrior;
            }

            $epsilon = max($this->smoothing * $maxVariance, $minEpsilon);

            foreach ($this->variances as &$variances) {
                foreach ($variances as &$variance) {
                    $variance += $epsilon;
                }
            }

            $loss /= $n;

            $lossChange = abs($loss - $prevLoss);

            $this->losses[$epoch] = $loss;

            if ($this->logger) {
                $lossDirection = $loss < $prevLoss ? '↓' : '↑';

                $message = "Epoch: $epoch, "
                    . "Loss: $loss, "
                    . "Loss Change: {$lossDirection}{$lossChange}";

                $this->logger->info($message);
            }

            if ($loss <= 0.0) {
                break;
            }

            if ($lossChange < $this->minChange) {
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
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->means) or empty($this->variances)) {
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
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->means) or empty($this->variances)) {
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
     * @return array<int,float>
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
     * Initialize the gaussian components by calculating the means and variances of k initial cluster centroids generated by the seeder.
     *
     * @param Dataset $dataset
     */
    protected function initialize(Dataset $dataset) : void
    {
        $this->logPriors = $this->means = $this->variances = [];

        $kernel = new Euclidean();

        $n = $dataset->numSamples();

        $maxVariance = 0.0;

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

        foreach ($clusters as $cluster => $samples) {
            $means = $variances = [];

            $features = array_transpose($samples);

            foreach ($features as $values) {
                [$mean, $variance] = Stats::meanVar($values);

                $means[] = $mean;
                $variances[] = $variance;
            }

            $maxVariance = max($maxVariance, ...$variances);

            $logPrior = log(count($samples) / $n);

            $this->means[$cluster] = $means;
            $this->variances[$cluster] = $variances;
            $this->logPriors[$cluster] = $logPrior;
        }

        $epsilon = max($this->smoothing * $maxVariance, CPU::epsilon());

        foreach ($this->variances as &$variances) {
            foreach ($variances as &$variance) {
                $variance += $epsilon;
            }
        }
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['losses']);

        return $properties;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Gaussian Mixture (' . Params::stringify($this->params()) . ')';
    }
}
