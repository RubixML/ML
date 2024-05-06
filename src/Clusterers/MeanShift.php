<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Clusterers\Seeders\Random;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function Rubix\ML\array_transpose;
use function is_nan;
use function array_map;
use function round;
use function exp;
use function get_object_vars;

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
    protected float $radius;

    /**
     * The precomputed denominator of the weight calculation.
     *
     * @var float
     */
    protected float $delta;

    /**
     * The ratio of samples from the training set to use as initial centroids.
     *
     * @var float
     */
    protected float $ratio;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * The minimum shift in the position of the centroids necessary to continue training.
     *
     * @var float
     */
    protected float $minShift;

    /**
     * The spatial tree used to run range searches.
     *
     * @var Spatial
     */
    protected Spatial $tree;

    /**
     * The cluster centroid seeder.
     *
     * @var Seeder
     */
    protected Seeder $seeder;

    /**
     * The computed centroid vectors of the training data.
     *
     * @var list<(int|float)[]>
     */
    protected array $centroids = [
        //
    ];

    /**
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * Estimate the radius of a cluster that encompasses a certain percentage of
     * the training samples.
     *
     * > **Note**: Since radius estimation scales quadratically in the number of
     * samples, for large datasets you can speed up the process by running it on
     * a smaller subset of the training data.
     *
     * @param Dataset $dataset
     * @param float $percentile
     * @param Distance|null $kernel
     * @throws InvalidArgumentException
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
     * @param Spatial|null $tree
     * @param Seeder|null $seeder
     * @throws InvalidArgumentException
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

        if ($epochs < 0) {
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
            'radius' => $this->radius,
            'ratio' => $this->ratio,
            'epochs' => $this->epochs,
            'min shift' => $this->minShift,
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
     * @return list<(int|float)[]>
     */
    public function centroids() : array
    {
        return $this->centroids;
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
     * Return the amount of centroid shift at each epoch of training.
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

        $n = $dataset->numSamples();

        $labels = range(0, $n - 1);

        $dataset = Labeled::quick($dataset->samples(), $labels);

        $k = max(self::MIN_SEEDS, (int) round($this->ratio * $n));

        /** @var list<list<int|float>> $centroids */
        $centroids = $this->seeder->seed($dataset, $k);

        $this->tree->grow($dataset);

        $this->losses = [];

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

            $loss /= $n;

            $this->losses[$epoch] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch: $epoch, Shift: $loss");
            }

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
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
     * @param Dataset $dataset
     * @throws RuntimeException
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
     * @param Dataset $dataset
     * @throws RuntimeException
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
     * @param list<(int|float)[]> $current
     * @param list<(int|float)[]> $previous
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
        return 'Mean Shift (' . Params::stringify($this->params()) . ')';
    }
}
