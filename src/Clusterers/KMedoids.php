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
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Kernels\Distance\Distance;
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

use function count;
use function is_nan;
use function array_fill;
use function array_map;
use function get_object_vars;

use const Rubix\ML\EPSILON;

/**
 * K Medoids
 *
 * A robust centroid-based hard clustering algorithm that uses actual data points
 * (medoids) as cluster centers instead of computed means. K Medoids is more resistant
 * to outliers than K Means and is suitable for clustering with arbitrary distance
 * metrics. The algorithm minimizes the sum of dissimilarities between samples and
 * their nearest medoid using the Partitioning Around Medoids (PAM) approach.
 *
 * References:
 * [1] L. Kaufman et al. (1987). Clustering by means of Medoids.
 * [2] H. S. Park et al. (2009). A simple and fast algorithm for K-medoids clustering.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Cheribet Cherif Chouaib
 */
class KMedoids implements Estimator, Learner, Online, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The target number of clusters.
     *
     * @var int<0,max>
     */
    protected int $k;

    /**
     * The size of each mini batch in samples.
     *
     * @var positive-int
     */
    protected int $batchSize;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * The minimum change in the inertia for training to continue.
     *
     * @var float
     */
    protected float $minChange;

    /**
     * The number of epochs without improvement in the training loss to wait before considering an early stop.
     *
     * @var int
     */
    protected int $window;

    /**
     * The distance function to use when computing the distances.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The cluster medoid seeder.
     *
     * @var Seeder
     */
    protected Seeder $seeder;

    /**
     * The medoid vectors of the training data (actual data points).
     *
     * @var list<list<int|float>>
     */
    protected array $medoids = [
        //
    ];

    /**
     * The complete dataset samples stored for medoid updates.
     *
     * @var list<list<int|float>>
     */
    protected array $samples = [
        //
    ];

    /**
     * The number of training samples contained within each cluster.
     *
     * @var int[]
     */
    protected array $sizes = [
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
     * @param int $batchSize
     * @param int $epochs
     * @param float $minChange
     * @param int $window
     * @param Distance|null $kernel
     * @param Seeder|null $seeder
     * @throws InvalidArgumentException
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

        if ($epochs < 0) {
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
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::clusterer();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @return list<DataType>
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
            'batch size' => $this->batchSize,
            'epochs' => $this->epochs,
            'min change' => $this->minChange,
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
        return !empty($this->medoids);
    }

    /**
     * Return the computed cluster medoids of the training data.
     *
     * @return list<list<int|float>>
     */
    public function medoids() : array
    {
        return $this->medoids;
    }

    /**
     * Return the number of training samples each medoid is responsible for.
     *
     * @return int[]
     */
    public function sizes() : array
    {
        return $this->sizes;
    }

    /**
     * Return an iterable progress table with the steps from the last training session.
     *
     * @return Generator<mixed[]>
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
     * Return the loss for each epoch from the last training session.
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

        /** @var list<list<int|float>> $seeds */
        $seeds = $this->seeder->seed($dataset, $this->k);

        $this->medoids = $seeds;
        $this->samples = $dataset->samples();

        $sizes = array_fill(0, $this->k, 0);
        $sizes[0] = $dataset->numSamples();

        $this->sizes = $sizes;

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param Dataset $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if (empty($this->medoids) or empty($this->sizes)) {
            $this->train($dataset);

            return;
        }

        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new DatasetHasDimensionality($dataset, count(current($this->medoids))),
        ])->check();

        // Merge new samples with existing samples
        $newSamples = $dataset->samples();
        $this->samples = array_merge($this->samples, $newSamples);

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        $allSamples = $this->samples;
        $labels = array_fill(0, count($allSamples), 0);

        $fullDataset = Labeled::quick($allSamples, $labels);

        $prevLoss = $bestLoss = INF;
        $numWorseEpochs = 0;

        $this->losses = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            // Assign all samples to nearest medoid
            $assignments = array_map([$this, 'predictSample'], $allSamples);

            // Group samples by cluster
            $clusters = array_fill(0, $this->k, []);
            
            foreach ($assignments as $i => $cluster) {
                $clusters[$cluster][] = $i;
            }

            // Update medoids - find the sample that minimizes total distance within each cluster
            foreach ($clusters as $cluster => $indices) {
                if (empty($indices)) {
                    continue;
                }

                $bestMedoidIndex = $indices[0];
                $bestTotalDistance = INF;

                // Try each point in the cluster as a potential medoid
                foreach ($indices as $candidateIndex) {
                    $totalDistance = 0.0;

                    foreach ($indices as $pointIndex) {
                        $totalDistance += $this->kernel->compute(
                            $allSamples[$candidateIndex],
                            $allSamples[$pointIndex]
                        );
                    }

                    if ($totalDistance < $bestTotalDistance) {
                        $bestTotalDistance = $totalDistance;
                        $bestMedoidIndex = $candidateIndex;
                    }
                }

                $this->medoids[$cluster] = $allSamples[$bestMedoidIndex];
                $this->sizes[$cluster] = count($indices);
            }

            // Calculate loss (total cost)
            $loss = $this->inertia($allSamples, $assignments);
            $loss /= count($allSamples);

            $lossChange = abs($prevLoss - $loss);

            $this->losses[$epoch] = $loss;

            if ($this->logger) {
                $lossDirection = $loss < $prevLoss ? '↓' : '↑';

                $message = "Epoch: $epoch, "
                    . "Inertia: $loss, "
                    . "Loss Change: {$lossDirection}{$lossChange}";

                $this->logger->info($message);
            }

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            if ($loss <= 0.0) {
                break;
            }

            if ($lossChange < $this->minChange) {
                break;
            }

            if ($loss < $bestLoss) {
                $bestLoss = $loss;

                $numWorseEpochs = 0;
            } else {
                ++$numWorseEpochs;
            }

            if ($numWorseEpochs >= $this->window) {
                break;
            }

            $prevLoss = $loss;
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Cluster the dataset by assigning a label to each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->medoids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->medoids)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Label a given sample based on its distance from a particular medoid.
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

        foreach ($this->medoids as $cluster => $medoid) {
            $distance = $this->kernel->compute($sample, $medoid);

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
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->medoids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->medoids)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the k medoids.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        $distances = $dist = [];

        foreach ($this->medoids as $medoid) {
            $distances[] = $this->kernel->compute($sample, $medoid) ?: EPSILON;
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
     * medoid.
     *
     * @param list<list<int|float>> $samples
     * @param list<int> $labels
     * @return float
     */
    protected function inertia(array $samples, array $labels) : float
    {
        $inertia = 0.0;

        foreach ($samples as $i => $sample) {
            $medoid = $this->medoids[$labels[$i]];

            $inertia += $this->kernel->compute($sample, $medoid);
        }

        return $inertia;
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['losses'], $properties['logger']);

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
        return 'K Medoids (' . Params::stringify($this->params()) . ')';
    }
}
