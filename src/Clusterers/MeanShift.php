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
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Clusterers\Seeders\Random;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\array_transpose;
use function is_nan;

use const Rubix\ML\EPSILON;

/**
 * Mean Shift
 *
 * A hierarchical clustering algorithm that uses peak finding to locate the candidate centroids of a
 * training set given a radius constraint. Near-duplicate candidates are merged together in a final
 * post-processing step.
 *
 * References:
 * [1] M. A. Carreira-Perpinan et al. (2015). A Review of Mean-shift Algorithms for Clustering.
 * [2] D. Comaniciu et al. (2012). Mean Shift: A Robust Approach Toward Feature Space Analysis.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MeanShift implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The minimum number of initial centroids.
     *
     * @var int
     */
    protected const MIN_SEEDS = 20;

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
     * The ratio of samples from the training set to use as initial centroids.
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
     * The minimum shift in the position of the centroids necessary to continue training.
     *
     * @var float
     */
    protected $minShift;

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
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected $steps;

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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public static function estimateRadius(
        Dataset $dataset,
        float $percentile = 30.0,
        ?Distance $kernel = null
    ) : float {
        if ($percentile < 0.0 or $percentile > 100.0) {
            throw new InvalidArgumentException('Percentile must be'
                . " between 0 and 100, $percentile given.");
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

        return Stats::quantile($distances, $percentile / 100.0);
    }

    /**
     * @param float $radius
     * @param float $ratio
     * @param int $epochs
     * @param float $minShift
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @param \Rubix\ML\Clusterers\Seeders\Seeder|null $seeder
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        float $radius,
        float $ratio = 0.1,
        int $epochs = 100,
        float $minShift = 1e-4,
        ?Spatial $tree = null,
        ?Seeder $seeder = null
    ) {
        if ($radius <= 0.0) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        if ($ratio <= 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minShift < 0.0) {
            throw new InvalidArgumentException('Minimum shift must be'
                . " greater than 0, $minShift given.");
        }

        $this->radius = $radius;
        $this->delta = 2.0 * $radius ** 2;
        $this->ratio = $ratio;
        $this->epochs = $epochs;
        $this->minShift = $minShift;
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
            'radius' => $this->radius,
            'ratio' => $this->ratio,
            'epochs' => $this->epochs,
            'min_shift' => $this->minShift,
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

        $labels = range(0, $n - 1);

        $dataset = Labeled::quick($dataset->samples(), $labels);

        $k = max(self::MIN_SEEDS, (int) round($this->ratio * $n));

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

            if ($loss < $this->minShift) {
                break;
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
            $distance = $this->tree->kernel()->compute($sample, $centroid);

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
        if (empty($this->centroids)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->centroids)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the centroids.
     *
     * @param list<int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        $distances = $dist = [];

        foreach ($this->centroids as $centroid) {
            $distances[] = $this->tree->kernel()->compute($sample, $centroid) ?: EPSILON;
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
     * Calculate the amount of centroid shift from the previous epoch.
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

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Mean Shift (' . Params::stringify($this->params()) . ')';
    }
}
