<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
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

use function count;
use function is_nan;

use const Rubix\ML\EPSILON;

/**
 * K Means
 *
 * A fast online centroid-based hard clustering algorithm capable of grouping linearly
 * separable data points given some prior knowledge of the target number of clusters
 * (defined by *k*). K Means is trained using adaptive Mini Batch Gradient Descent and
 * minimizes the inertia cost function. Inertia is defined as the average sum of distances
 * between each sample and its nearest cluster centroid.
 *
 * References:
 * [1] D. Sculley. (2010). Web-Scale K-Means Clustering.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KMeans implements Estimator, Learner, Online, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The target number of clusters.
     *
     * @var int
     */
    protected $k;

    /**
     * The size of each mini batch in samples.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The maximum number of iterations to run until the algorithm
     * terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The minimum change in the inertia for training to continue.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The number of epochs without improvement in the training loss to wait
     * before considering an early stop.
     *
     * @var int
     */
    protected $window;

    /**
     * The distance function to use when computing the distances.
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
     * The number of training samples contained within each cluster
     * centroid.
     *
     * @var int[]
     */
    protected $sizes = [
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
     * @param int $batchSize
     * @param int $epochs
     * @param float $minChange
     * @param int $window
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param \Rubix\ML\Clusterers\Seeders\Seeder|null $seeder
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        int $k,
        int $batchSize = 128,
        int $epochs = 1000,
        float $minChange = 1e-4,
        int $window = 5,
        ?Distance $kernel = null,
        ?Seeder $seeder = null
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be greater'
                . " than 0, $k given.");
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size must be'
                . " greater than 0, $batchSize given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be'
                . " greater than 0, $window given.");
        }

        $this->k = $k;
        $this->batchSize = $batchSize;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->window = $window;
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
            'k' => $this->k,
            'batch_size' => $this->batchSize,
            'epochs' => $this->epochs,
            'min_change' => $this->minChange,
            'window' => $this->window,
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
     * Return the number of training samples each centroid is responsible for.
     *
     * @return int[]
     */
    public function sizes() : array
    {
        return $this->sizes;
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

        $this->centroids = $this->seeder->seed($dataset, $this->k);

        $sizes = array_fill(0, $this->k, 0);
        $sizes[0] = $dataset->numRows();

        $this->sizes = $sizes;

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if (empty($this->centroids) or empty($this->sizes)) {
            $this->train($dataset);

            return;
        }

        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new DatasetHasDimensionality($dataset, count(current($this->centroids))),
        ])->check();

        if ($this->logger) {
            $this->logger->info("$this initialized");
        }

        $labels = array_fill(0, $dataset->numRows(), 0);

        $dataset = Labeled::quick($dataset->samples(), $labels);

        $prevLoss = $bestLoss = INF;
        $delta = 0;

        $this->steps = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $batches = $dataset->randomize()->batch($this->batchSize);

            $loss = 0.0;

            foreach ($batches as $i => &$batch) {
                $assignments = array_map([$this, 'predictSample'], $batch->samples());

                $labels = $batch->labels();

                foreach ($assignments as $j => $cluster) {
                    $expected = $labels[$j];

                    if ($cluster !== $expected) {
                        $labels[$j] = $cluster;

                        --$this->sizes[$expected];
                        ++$this->sizes[$cluster];
                    }
                }

                $batch = Labeled::quick($batch->samples(), $labels);

                $loss += $this->inertia($batch->samples(), $labels);

                if (is_nan($loss)) {
                    if ($this->logger) {
                        $this->logger->info('Numerical instability detected');
                    }

                    break;
                }

                foreach ($batch->stratify() as $cluster => $stratum) {
                    $centroid = &$this->centroids[$cluster];

                    $means = array_map([Stats::class, 'mean'], $stratum->columns());

                    $weight = 1.0 / (1 + $this->sizes[$cluster]);

                    foreach ($centroid as $i => &$mean) {
                        $mean = (1.0 - $weight) * $mean + $weight * $means[$i];
                    }
                }
            }

            $loss /= $dataset->numRows();

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - Inertia: $loss");
            }

            if ($loss <= 0.0) {
                break;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break;
            }

            if ($loss < $bestLoss) {
                $bestLoss = $loss;

                $delta = 0;
            } else {
                ++$delta;
            }

            if ($delta >= $this->window) {
                break;
            }

            $dataset = Labeled::stack($batches);

            $prevLoss = $loss;
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Cluster the dataset by assigning a label to each sample.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->centroids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->centroids)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return int
     */
    public function predictSample(array $sample) : int
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

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->centroids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->centroids)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the k centroids.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        $distances = $dist = [];

        foreach ($this->centroids as $centroid) {
            $distances[] = $this->kernel->compute($sample, $centroid) ?: EPSILON;
        }

        foreach ($distances as $distanceA) {
            $sigma = 0.0;

            foreach ($distances as $distanceB) {
                $sigma += $distanceA / $distanceB;
            }

            $dist[] = 1.0 / $sigma;
        }

        return $dist;
    }

    /**
     * Calculate the average sum of distances between all samples and their closest
     * centroid.
     *
     * @param list<list<int|float>> $samples
     * @param list<int> $labels
     * @return float
     */
    protected function inertia(array $samples, array $labels) : float
    {
        $inertia = 0.0;

        foreach ($samples as $i => $sample) {
            $centroid = $this->centroids[$labels[$i]];

            $inertia += $this->kernel->compute($sample, $centroid);
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
        return 'K Means (' . Params::stringify($this->params()) . ')';
    }
}
