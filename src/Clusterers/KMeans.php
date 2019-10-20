<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\Tensor\Matrix;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * K Means
 *
 * A fast online centroid-based hard clustering algorithm capable of clustering
 * linearly separable data points given some prior knowledge of the target number
 * of clusters (defined by *k*). K Means is trained using adaptive mini batch
 * Gradient Descent and minimizes the inertia cost function. Inertia is defined
 * as the average sum of distances between each sample and its nearest cluster
 * centroid.
 *
 * References:
 * [1] D. Sculley. (2010). Web-Scale K-Means Clustering.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KMeans implements Estimator, Learner, Online, Probabilistic, Persistable, Verbose
{
    use PredictsSingle, LoggerAware;

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
     * The inertia at each epoch from the last round of training.
     *
     * @var float[]
     */
    protected $steps = [
        //
    ];

    /**
     * @param int $k
     * @param int $batchSize
     * @param int $epochs
     * @param float $minChange
     * @param int $window
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param \Rubix\ML\Clusterers\Seeders\Seeder|null $seeder
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $k,
        int $batchSize = 100,
        int $epochs = 1000,
        float $minChange = 1e-4,
        int $window = 5,
        ?Distance $kernel = null,
        ?Seeder $seeder = null
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('Must target at least 1'
                . " cluster, $k given.");
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size must be greater'
                . " than 0, $batchSize given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train'
                . " for at least 1 epoch, $epochs given.");
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException('Min change cannot be less'
                . " than 1, $minChange given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be at least 1'
                . " epoch, $window given.");
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
     * Return the number of training samples each centroid is responsible for.
     *
     * @return int[]
     */
    public function sizes() : array
    {
        return $this->sizes;
    }

    /**
     * Return the value of the inertial function at each epoch from the last
     * round of training.
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
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->centroids = $this->seeder->seed($dataset, $this->k);

        $sizes = array_fill(0, $this->k, 0);
        $sizes[0] = $dataset->numRows();

        $this->sizes = $sizes;

        $this->steps = [];

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function partial(Dataset $dataset) : void
    {
        if (empty($this->centroids) or empty($this->sizes)) {
            $this->train($dataset);

            return;
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner init ' . Params::stringify([
                'k' => $this->k,
                'batch_size' => $this->batchSize,
                'epochs' => $this->epochs,
                'min_change' => $this->minChange,
                'window' => $this->window,
                'kernel' => $this->kernel,
                'seeder' => $this->seeder,
            ]));
        }

        $samples = $dataset->samples();

        $clusters = array_fill(0, $dataset->numRows(), 0);

        $order = range(0, $dataset->numRows() - 1);

        $prevLoss = $bestLoss = INF;
        $nu = 0;

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            shuffle($order);

            array_multisort($order, $samples, $clusters);

            $sBatches = array_chunk($samples, $this->batchSize, true);
            $lBatches = array_chunk($clusters, $this->batchSize, true);

            foreach ($sBatches as $i => $batch) {
                $assignments = array_map([self::class, 'assign'], $batch);

                $labels = $lBatches[$i];

                foreach ($assignments as $j => $cluster) {
                    $expected = $labels[$j];

                    if ($cluster !== $expected) {
                        $clusters[$j] = $cluster;
                        $labels[$j] = $cluster;

                        $this->sizes[$expected]--;
                        $this->sizes[$cluster]++;
                    }
                }

                $strata = Labeled::quick($batch, $labels)->stratify();

                foreach ($strata as $cluster => $stratum) {
                    $centroid = $this->centroids[$cluster];

                    $step = Matrix::quick($stratum->samples())
                        ->transpose()
                        ->mean()
                        ->asArray();

                    $weight = 1. / ($this->sizes[$cluster] ?: EPSILON);

                    foreach ($centroid as $i => &$mean) {
                        $mean = (1. - $weight) * $mean + $weight * $step[$i];
                    }

                    $this->centroids[$cluster] = $centroid;
                }
            }

            $loss = $this->inertia($samples);

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch loss=$loss");
            }

            if ($loss < $bestLoss) {
                $bestLoss = $loss;
                
                $nu = 0;
            } else {
                $nu++;
            }

            if (is_nan($loss) or $loss < EPSILON) {
                break 1;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break 1;
            }

            if ($nu >= $this->window) {
                break 1;
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->centroids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'assign'], $dataset->samples());
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
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param array $sample
     * @return int
     */
    protected function assign(array $sample) : int
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
     * Return the membership of a sample to each of the k centroids.
     *
     * @param array $sample
     * @return array
     */
    protected function membership(array $sample) : array
    {
        $membership = $distances = [];

        foreach ($this->centroids as $centroid) {
            $distances[] = $this->kernel->compute($sample, $centroid);
        }

        $total = array_sum($distances) ?: EPSILON;

        foreach ($distances as $distance) {
            $membership[] = $distance / $total;
        }

        return $membership;
    }

    /**
     * Calculate the average sum of distances between all samples and their closest
     * centroid.
     *
     * @param array $samples
     * @return float
     */
    protected function inertia(array $samples) : float
    {
        if (empty($samples)) {
            return 0.;
        }

        $inertia = 0.;

        foreach ($samples as $sample) {
            foreach ($this->centroids as $centroid) {
                $inertia += $this->kernel->compute($sample, $centroid);
            }
        }

        return $inertia / count($samples);
    }
}
