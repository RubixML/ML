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
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function Rubix\ML\argmin;
use function count;
use function array_map;
use function array_search;
use function array_merge;
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
 * remains.
 *
 * The increase in WCSS incurred by each merge — the *merge cost* — is tracked
 * with the **Lance-Williams** recurrence, which allows the full cost matrix to
 * be updated in O(n) per round instead of being recomputed from scratch:
 *
 *     δ(C, A∪B) = ((|C|+|A|)·δ(C,A) + (|C|+|B|)·δ(C,B) − |C|·δ(A,B))
 *                 / (|C|+|A|+|B|)
 *
 * For the squared-Euclidean kernel this recovers the classic Ward minimum-
 * variance rule exactly (the ½·‖a − b‖² seed is the within-cluster sum of
 * squares of the two-point cluster, and the recurrence exactly tracks the
 * increase in WCSS). Because this objective only holds for the squared-Euclidean
 * metric, the distance kernel is restricted to the Euclidean family. The
 * algorithm is deterministic: given the same data and the same hyper-parameters
 * the resulting clustering is always identical.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Ward implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The list of allowed distance kernels for the Ward algorithm.
     */
    public const array ALLOWED_KERNELS = [
        Euclidean::class,
        SafeEuclidean::class,
    ];

    /**
     * The target number of clusters.
     *
     * @var positive-int
     */
    protected int $k;

    /**
     * The distance kernel used to compute the pairwise distance between samples.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The computed cluster centroids of the training data.
     *
     * @var list<list<string|int|float>>
     */
    protected array $centroids = [
        //
    ];

    /**
     * The number of training samples assigned to each cluster.
     *
     * @var int[]
     */
    protected array $sizes = [
        //
    ];

    /**
     * The cluster assignment of each training sample (index = sample offset).
     *
     * @var list<int>
     */
    protected array $labels = [
        //
    ];

    /**
     * The cost — the increase in the total within-cluster sum of squares —
     * incurred by each merge from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param int $k
     * @param Distance|null $kernel
     * @throws InvalidArgumentException
     */
    public function __construct(int $k, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be greater'
                . " than 0, $k given.");
        }

        if (isset($kernel) and !in_array($kernel::class, self::ALLOWED_KERNELS)) {
            throw new InvalidArgumentException('Kernel must be one of '
                . implode(', ', self::ALLOWED_KERNELS) . ", $kernel given.");
        }

        $this->k = $k;
        $this->kernel = $kernel ?? new Euclidean();
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
            'kernel' => $this->kernel,
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
     * @return list<list<string|int|float>>
     */
    public function centroids() : array
    {
        return $this->centroids;
    }

    /**
     * Return the number of training samples each cluster is responsible for.
     *
     * @return int[]
     */
    public function sizes() : array
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

        $parent = range(0, $n - 1);
        $sizes = array_fill(0, $n, 1);
        $memberships = array_map(fn ($i) => [$i], range(0, $n - 1));

        $costs = $this->initializeCosts($dataset);

        $this->losses = [];

        for ($epoch = 1; $epoch <= $maxMerges; ++$epoch) {
            $i = 0;
            $j = 1;
            $min = INF;

            for ($a = 0; $a < $n; ++$a) {
                for ($b = $a + 1; $b < $n; ++$b) {
                    if ($costs[$a][$b] < $min) {
                        $min = $costs[$a][$b];

                        $i = $a;
                        $j = $b;
                    }
                }
            }

            if ($this->logger) {
                $this->logger->info("Epoch: $epoch, Cost: $min");
            }

            if (is_nan($min)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            $this->losses[$epoch] = $min;

            $a = $i;
            $b = $j;
            $nA = $sizes[$a];
            $nB = $sizes[$b];
            $dAB = $costs[$a][$b];

            $memberships[$a] = array_merge($memberships[$a], $memberships[$b]);
            $parent[$b] = $a;

            for ($c = 0; $c < $n; ++$c) {
                if ($c === $a or $c === $b) {
                    continue;
                }

                $nC = $sizes[$c];

                $newCost = (($nA + $nC) * $costs[$a][$c] + ($nB + $nC) * $costs[$b][$c] - $nC * $dAB)
                    / ($nA + $nB + $nC);

                $costs[$a][$c] = $newCost;
                $costs[$c][$a] = $newCost;

                $costs[$b][$c] = INF;
                $costs[$c][$b] = INF;
            }

            $costs[$a][$b] = INF;
            $costs[$b][$a] = INF;

            $sizes[$a] += $sizes[$b];
            $sizes[$b] = 0;
        }

        $roots = [];

        for ($root = 0; $root < $n; ++$root) {
            if ($parent[$root] !== $root) {
                continue;
            }

            if ($sizes[$root] > 0) {
                $roots[] = $root;
            }
        }

        $count = count($roots);

        $labels = [];

        for ($i = 0; $i < $n; ++$i) {
            $root = $i;

            while ($parent[$root] !== $root) {
                $root = $parent[$root];
            }

            $cluster = array_search($root, $roots, true);

            if ($cluster === false) {
                throw new RuntimeException('Failed to resolve cluster for a sample.');
            }

            $labels[$i] = $cluster;
        }

        $sizes = array_fill(0, $count, 0);

        foreach ($labels as $i => $cluster) {
            ++$sizes[$cluster];
        }

        $centroids = [];

        foreach ($roots as $root) {
            $cluster = array_search($root, $roots, true);

            $sum = array_fill(0, count($dataset->sample(0)), 0.0);

            foreach ($memberships[$root] as $idx) {
                $sample = $dataset->sample($idx);

                foreach ($sample as $x => $value) {
                    $sum[$x] += $value;
                }
            }

            $size = $sizes[$cluster];

            $centroids[$cluster] = array_map(fn ($v) => $v / $size, $sum);
        }

        $this->centroids = $centroids;
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
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->centroids)))->check();

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
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->centroids)))->check();

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

        foreach ($this->centroids as $centroid) {
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

        foreach ($this->centroids as $centroid) {
            $distances[] = $this->kernel->compute($sample, $centroid);
        }

        return $distances;
    }

    /**
     * Compute the initial merge-cost matrix between every pair of singleton
     * clusters using the distance kernel.
     *
     * The cost of merging two singletons is ½·‖a − b‖²; we square the kernel
     * distance because the Ward recurrence is derived from the squared-distance
     * WCSS relation.
     *
     * @param Dataset $dataset
     * @return list<list<float>>
     */
    protected function initializeCosts(Dataset $dataset) : array
    {
        $n = $dataset->numSamples();

        $costs = array_fill(0, $n, array_fill(0, $n, 0.0));

        for ($i = 0; $i < $n; ++$i) {
            for ($j = $i + 1; $j < $n; ++$j) {
                $distance = $this->kernel->compute($dataset->sample($i), $dataset->sample($j));
                $cost = 0.5 * $distance * $distance;

                $costs[$i][$j] = $cost;
                $costs[$j][$i] = $cost;
            }
        }

        return $costs;
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
