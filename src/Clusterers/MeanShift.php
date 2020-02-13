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
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Clusterers\Seeders\Random;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\array_transpose;

use const Rubix\ML\EPSILON;

/**
 * Mean Shift
 *
 * A hierarchical clustering algorithm that uses peak finding to locate the candidate
 * centroids of a training set given a radius constraint. Near-duplicate candidates
 * are merged together in a final post-processing step.
 *
 * References:
 * [1] M. A. Carreira-Perpinan et al. (2015). A Review of Mean-shift Algorithms
 * for Clustering.
 * [2] D. Comaniciu et al. (2012). Mean Shift: A Robust Approach Toward Feature
 * Space Analysis.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MeanShift implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use PredictsSingle, ProbaSingle, LoggerAware;

    /**
     * The minimum number of centroid seeds.
     *
     * @var int
     */
    protected const MIN_SEEDS = 10;

    /**
     * The maximum distance between two points to be considered neighbors.
     *
     * @var float
     */
    protected $radius;

    /**
     * The precomputed denominator of the weight calculation.
     *
     * @var float
     */
    protected $delta;

    /**
     * The ratio of samples from the training set to seed the algorithm with.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The minimum change in the centroids necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The spatial tree used to run range searches.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

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
     * The amount of centroid shift during each epoch of training.
     *
     * @var float[]
     */
    protected $steps = [
        //
    ];

    /**
     * Estimate the radius of a cluster that encompasses a certain percentage of
     * the training samples.
     *
     * > **Note**: Since radius estimation scales quadratically in the number of
     * samples, for large datasets you can speed up the process by running it on
     * a smaller subset of the training data.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param float $percentile
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function estimateRadius(
        Dataset $dataset,
        float $percentile = 30.0,
        ?Distance $kernel = null
    ) : float {
        if ($percentile < 0.0 or $percentile > 100.0) {
            throw new InvalidArgumentException('Percentile must be between'
                . " 0 and 100, $percentile given.");
        }

        $kernel = $kernel ?? new Euclidean();

        $samples = $dataset->samples();

        $distances = [];

        foreach ($samples as $i => $sampleA) {
            foreach ($samples as $j => $sampleB) {
                if ($i !== $j) {
                    $distances[] = $kernel->compute($sampleA, $sampleB);
                }
            }
        }

        return Stats::percentile($distances, $percentile);
    }

    /**
     * @param float $radius
     * @param float $ratio
     * @param int $epochs
     * @param float $minChange
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @param \Rubix\ML\Clusterers\Seeders\Seeder|null $seeder
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $radius,
        float $ratio = 0.1,
        int $epochs = 100,
        float $minChange = 1e-4,
        ?Spatial $tree = null,
        ?Seeder $seeder = null
    ) {
        if ($radius <= 0.0) {
            throw new InvalidArgumentException('Cluster radius must be'
                . " greater than 0, $radius given.");
        }

        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0.01 and 1, $ratio given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Learner must train for at'
                . " least 1 epoch, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . " than 0, $minChange given.");
        }

        $this->radius = $radius;
        $this->delta = 2.0 * $radius ** 2;
        $this->ratio = $ratio;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->tree = $tree ?? new BallTree();
        $this->seeder = $seeder ?? new Random();
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
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
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
            'radius' => $this->radius,
            'ratio' => $this->ratio,
            'epochs' => $this->epochs,
            'min_change' => $this->minChange,
            'tree' => $this->tree,
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
     * Return the amount of centroid shift at each epoch of training.
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
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner init ' . Params::stringify($this->params()));

            $this->logger->info('Training started');
        }

        $n = $dataset->numRows();

        $dataset = Labeled::quick($dataset->samples(), range(0, $n - 1));

        $k = max(min(self::MIN_SEEDS, $n), (int) round($this->ratio * $n));

        $centroids = $this->seeder->seed($dataset, $k);

        $this->tree->grow($dataset);

        $this->steps = [];

        $previous = $centroids;
 
        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            foreach ($centroids as $i => &$centroidA) {
                [$samples, $indices, $distances] = $this->tree->range($centroidA, $this->radius);

                $means = array_map([Stats::class, 'mean'], array_transpose($samples));

                $mu2 = Stats::mean($distances) ** 2;

                $weight = exp(-$mu2 / $this->delta);

                foreach ($centroidA as $column => &$mean) {
                    $mean = ($weight * $means[$column]) / $weight;
                }

                foreach ($centroids as $j => $centroidB) {
                    if ($i !== $j) {
                        $distance = $this->tree->kernel()->compute($centroidA, $centroidB);

                        if ($distance < $this->radius) {
                            unset($centroids[$j]);
                        }
                    }
                }
            }

            $loss = $this->shift($centroids, $previous);

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch loss=$loss");
            }

            if (is_nan($loss) or $loss < EPSILON) {
                break 1;
            }

            if ($loss < $this->minChange) {
                break 1;
            }

            $previous = $centroids;
        }

        $this->centroids = array_values($centroids);

        $this->tree->destroy();

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Cluster the dataset by assigning a label to each sample.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return int[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return array_map([self::class, 'assign'], $dataset->samples());
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return array_map([self::class, 'membership'], $dataset->samples());
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param (int|float)[] $sample
     * @return int
     */
    protected function assign(array $sample) : int
    {
        $bestDistance = INF;
        $bestCluster = -1;

        foreach ($this->centroids as $cluster => $centroid) {
            $distance = $this->tree->kernel()->compute($sample, $centroid);

            if ($distance < $bestDistance) {
                $bestDistance = $distance;
                $bestCluster = $cluster;
            }
        }

        return (int) $bestCluster;
    }

    /**
     * Return the membership of a sample to each of the centroids.
     *
     * @param (int|float)[] $sample
     * @return float[]
     */
    protected function membership(array $sample) : array
    {
        $membership = $distances = [];

        foreach ($this->centroids as $centroid) {
            $distances[] = $this->tree->kernel()->compute($sample, $centroid);
        }

        $total = array_sum($distances) ?: EPSILON;

        foreach ($distances as $distance) {
            $membership[] = $distance / $total;
        }

        return $membership;
    }

    /**
     * Calculate the magnitude (l1) of centroid shift from the previous epoch.
     *
     * @param array[] $current
     * @param array[] $previous
     * @return float
     */
    protected function shift(array $current, array $previous) : float
    {
        $shift = 0.0;

        foreach ($current as $cluster => $centroid) {
            $prevCentroid = $previous[$cluster];

            foreach ($centroid as $column => $mean) {
                $shift += abs($prevCentroid[$column] - $mean);
            }
        }

        return $shift;
    }
}
