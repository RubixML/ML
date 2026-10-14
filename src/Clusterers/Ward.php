<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\SquaredEuclidean;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use SplMinHeap;
use Generator;

use function Rubix\ML\argmin;
use function count;
use function array_map;
use function get_object_vars;
use function is_nan;

use const Rubix\ML\EPSILON;

/**
 * Ward
 *
 * An agglomerative (bottom-up) hierarchical clustering algorithm named after
 * *J. H. Ward, Jr.*. Unlike the iterative re-estimation schemes used by
 * [K Means](k-means.md) or [K Medoids](k-medoids.md), Ward begins with each
 * sample as its own singleton cluster and then *greedily* merges the pair of
 * clusters whose union causes the smallest increase in the total *within-cluster
 * sum of squares* (WCSS), repeating until the target number of clusters *k*
 * remains. Merging halts early if the cost of the next cheapest pair of
 * clusters exceeds the *maxDistance* threshold, which — defaulted to
 * `INF` — permits the full *n - k* merges.
 *
 * The increase in WCSS incurred by each merge — the *merge cost* — is tracked
 * with the Lance-Williams recurrence, which allows the full cost matrix to
 * be updated in O(n) per round instead of being recomputed from scratch:
 *
 *     δ(C, A∪B) = ((|C|+|A|)·δ(C,A) + (|C|+|B|)·δ(C,B) − |C|·δ(A,B))
 *                 / (|C|+|A|+|B|)
 *
 * The recurrence is only valid when the distance is the [squared Euclidean
 * distance](../kernels/distance/squared-euclidean.md), so the kernel —
 * defaulted to `SquaredEuclidean` — is restricted to that form.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Ward implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The target number of clusters.
     *
     * @var positive-int
     */
    protected int $k;

    /**
     * The maximum cost — the increase in the total within-cluster sum of
     * squares — a single merge may incur before training halts early.
     *
     * @var float
     */
    protected float $maxDistance;

    /**
     * The distance kernel used to compute the squared Euclidean distance
     * between samples — the cost the **Lance-Williams** recurrence is
     * derived for.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The computed cluster centroids of the training data.
     *
     * @var list<list<string|int|float>>|null
     */
    protected ?array $centroids = null;

    /**
     * The number of training samples assigned to each cluster.
     *
     * @var int[]|null
     */
    protected ?array $sizes = null;

    /**
     * The cost — the increase in the total within-cluster sum of squares —
     * incurred by each merge from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param int $k
     * @param float $maxDistance
     * @throws InvalidArgumentException
     */
    public function __construct(int $k, float $maxDistance = INF)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be greater'
                . " than 0, $k given.");
        }

        if ($maxDistance < 0.0) {
            throw new InvalidArgumentException('Max distance must be positive,'
                . " $maxDistance given.");
        }

        $this->k = $k;
        $this->maxDistance = $maxDistance;
        $this->kernel = new SquaredEuclidean();
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
        return [DataType::continuous()];
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
            'maxDistance' => $this->maxDistance,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->centroids !== null;
    }

    /**
     * Return the computed cluster centroids of the training data.
     *
     * @return list<list<string|int|float>>|null
     */
    public function centroids() : ?array
    {
        return $this->centroids;
    }

    /**
     * Return the number of training samples each cluster is responsible for.
     *
     * @return int[]|null
     */
    public function sizes() : ?array
    {
        return $this->sizes;
    }

    /**
     * Return an iterable progress table with the epoch steps from the last
     * training session.
     *
     * @return Generator<mixed[]>
     */
    public function progress() : Generator
    {
        if (!$this->losses) {
            return;
        }

        foreach ($this->losses as $epoch => $cost) {
            yield [
                'Epoch' => $epoch,
                'Loss' => $cost,
            ];
        }
    }

    /**
     * Return the merge cost — the increase in the total within-cluster sum of
     * squares — for each merge from the last training session.
     *
     * @return float[]|null
     */
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * Train the learner with a dataset by performing *n - k* agglomerative merges
     * using Ward's method with the Lance-Williams cost recurrence.
     *
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     * @throws RuntimeException
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        $n = $dataset->numSamples();

        if ($n < $this->k) {
            throw new InvalidArgumentException("Dataset must contain at least {$this->k}"
                . " samples, $n given.");
        }

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        $maxMerges = $n - $this->k;

        $samples = $dataset->samples();
        $numFeatures = $dataset->numFeatures();

        $cost = array_fill(0, $n * $n, 0.0);

        $heap = new SplMinHeap();

        for ($i = 0; $i < $n; ++$i) {
            for ($j = $i + 1; $j < $n; ++$j) {
                $distance = 0.5 * $this->kernel->compute($samples[$i], $samples[$j]);

                $cost[$i * $n + $j] = $distance;
                $cost[$j * $n + $i] = $distance;

                $heap->insert([$distance, $i, $j]);
            }
        }

        $parent = range(0, $n - 1);
        $sizes = array_fill(0, $n, 1);
        $sums = array_map(fn ($row) => array_map('floatval', $row), $samples);

        $this->losses = [];

        for ($epoch = 1; $epoch <= $maxMerges; ++$epoch) {
            $candidate = null;

            while ($heap->count() > 0) {
                [$distance, $i, $j] = $heap->extract();

                if ($parent[$i] !== $i or $parent[$j] !== $j) {
                    continue;
                }

                if ($distance !== $cost[$i * $n + $j]) {
                    continue;
                }

                $candidate = [$distance, $i, $j];

                break;
            }

            if ($candidate === null) {
                break;
            }

            [$distance, $a, $b] = $candidate;

            if ($distance > $this->maxDistance) {
                break;
            }

            $nA = $sizes[$a];
            $nB = $sizes[$b];

            if (is_nan($distance)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            if ($this->logger) {
                $this->logger->info("Epoch: $epoch, Cost: $distance");
            }

            $this->losses[$epoch] = $distance;

            $parent[$b] = $a;
            $sizes[$a] += $sizes[$b];
            $sizes[$b] = 0;

            for ($x = 0; $x < $numFeatures; ++$x) {
                $sums[$a][$x] += $sums[$b][$x];
            }

            for ($c = 0; $c < $n; ++$c) {
                if ($parent[$c] !== $c or $c === $a or $c === $b) {
                    continue;
                }

                $kAC = ($a < $c ? $a : $c) * $n + ($a < $c ? $c : $a);
                $kBC = ($b < $c ? $b : $c) * $n + ($b < $c ? $c : $b);
                $nC = $sizes[$c];

                $numerator = ($nA + $nC) * $cost[$kAC] + ($nB + $nC) * $cost[$kBC] - $nC * $distance;

                $newDistance = $numerator / ($nA + $nB + $nC);

                $lo = $a < $c ? $a : $c;
                $hi = $a < $c ? $c : $a;

                $cost[$lo * $n + $hi] = $newDistance;
                $cost[$hi * $n + $lo] = $newDistance;

                $heap->insert([$newDistance, $lo, $hi]);
            }
        }

        $roots = [];

        for ($root = 0; $root < $n; ++$root) {
            if ($parent[$root] === $root and $sizes[$root] > 0) {
                $roots[$root] = count($roots);
            }
        }

        $count = count($roots);

        $labels = [];

        for ($i = 0; $i < $n; ++$i) {
            $root = $i;

            while ($parent[$root] !== $root) {
                $root = $parent[$root];
            }

            $labels[$i] = $roots[$root];
        }

        $sizes = array_fill(0, $count, 0);

        foreach ($labels as $cluster) {
            ++$sizes[$cluster];
        }

        $centroids = [];

        foreach ($roots as $root => $cluster) {
            $size = $sizes[$cluster];

            $centroids[$cluster] = array_map(fn ($v) => $v / $size, $sums[$root]);
        }

        $this->centroids = array_values($centroids);
        $this->sizes = $sizes;

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
        $centroids = $this->centroids;

        if (!$centroids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($centroids)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Label a given sample based on its proximity to each cluster centroid.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        return argmin($this->centroidDistances($sample));
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
        $centroids = $this->centroids;

        if (!$centroids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($centroids)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the *k* cluster centroids.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        $distances = $proba = [];

        foreach ($this->centroids ?? [] as $centroid) {
            $distances[] = $this->kernel->compute($sample, $centroid) ?: EPSILON;
        }

        foreach ($distances as $distanceA) {
            $sigma = 0.0;

            foreach ($distances as $distanceB) {
                $sigma += $distanceA / $distanceB;
            }

            $proba[] = 1.0 / $sigma;
        }

        return $proba;
    }

    /**
     * Compute the distance from a sample to each cluster centroid.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return list<float>
     */
    protected function centroidDistances(array $sample) : array
    {
        $distances = [];

        foreach ($this->centroids ?? [] as $centroid) {
            $distances[] = $this->kernel->compute($sample, $centroid);
        }

        return $distances;
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
     * Restore the object from an associative array of serialized properties.
     *
     * @param mixed[] $properties
     */
    public function __unserialize(array $properties) : void
    {
        foreach ($properties as $property => $value) {
            $this->{$property} = $value;
        }
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
        return 'Ward (' . Params::stringify($this->params()) . ')';
    }
}
