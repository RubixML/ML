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
use Rubix\ML\Kernels\Distance\Symmetric;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function Rubix\ML\argmin;
use function count;
use function min;
use function range;
use function array_fill;
use function array_map;
use function get_object_vars;
use function is_nan;
use function usort;

use const Rubix\ML\EPSILON;

/**
 * SLINK
 *
 * An *agglomerative* (bottom-up) hierarchical clustering algorithm that groups
 * samples by the *single-link* (nearest neighbor) criterion, named after the
 * optimally efficient algorithm of *R. Sibson* [1]. Beginning with each sample
 * as its own singleton cluster, SLINK repeatedly merges the two clusters whose
 * closest pair of members is nearest, until the target number of clusters *k*
 * remains. Merging halts early if the distance of the next closest pair of
 * clusters exceeds the *maxDistance* threshold, which — defaulted to `INF` —
 * permits the full *n - k* merges.
 *
 * Rather than storing the pairwise distance matrix, SLINK maintains a compact
 * *pointer representation* of the dendrogram in two length-n arrays: a parent
 * pointer π and a merge height λ. Adding the *i*-th sample refreshes the
 * representation in O(i), giving a total running time of O(n²) with only O(n)
 * auxiliary space — optimal in both bounds. The k-cluster partition is
 * recovered by walking the merge sequence encoded by the pointer
 * representation and cutting it after *n - k* merges.
 *
 * Because single-link clusters are connected by chains of nearby samples, they
 * tend to be elongated and are therefore anchored by a *medoid* — the actual
 * training sample with the lowest total distance to the rest of its cluster —
 * rather than a mean vector, making predictions robust to the chaining effect
 * and valid for any symmetric distance kernel.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SLINK implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The target number of clusters.
     *
     * @var positive-int
     */
    protected int $k;

    /**
     * The maximum single-link distance a merge may incur before training
     * halts early.
     *
     * @var float
     */
    protected float $maxDistance;

    /**
     * The distance kernel used to compute the dissimilarity between samples.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The computed cluster medoids of the training data i.e. the actual samples
     * that anchor each cluster.
     *
     * @var list<list<string|int|float>>|null
     */
    protected ?array $medoids = null;

    /**
     * The number of training samples assigned to each cluster.
     *
     * @var int[]|null
     */
    protected ?array $sizes = null;

    /**
     * The single-link merge height recorded at each agglomeration round from
     * the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param int $k
     * @param float $maxDistance
     * @param Distance|null $kernel
     * @throws InvalidArgumentException
     */
    public function __construct(int $k, float $maxDistance = INF, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be greater'
                . " than 0, $k given.");
        }

        if ($maxDistance < 0.0) {
            throw new InvalidArgumentException('Max distance must be positive,'
                . " $maxDistance given.");
        }

        if (isset($kernel) and !$kernel instanceof Symmetric) {
            throw new InvalidArgumentException('Kernel must implement the Symmetric interface.');
        }

        $this->k = $k;
        $this->maxDistance = $maxDistance;
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
        return $this->kernel->compatibility();
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
            'max distance' => $this->maxDistance,
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
        return $this->medoids !== null;
    }

    /**
     * Return the computed cluster medoids of the training data i.e. the actual
     * samples that anchor each cluster.
     *
     * @return list<list<string|int|float>>|null
     */
    public function medoids() : ?array
    {
        return $this->medoids;
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

        foreach ($this->losses as $epoch => $distance) {
            yield [
                'Epoch' => $epoch,
                'Loss' => $distance,
            ];
        }
    }

    /**
     * Return the single-link merge height recorded at each agglomeration round
     * from the last training session.
     *
     * @return float[]|null
     */
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * Train the learner with a dataset by computing the pointer representation
     * of the single-link dendrogram and recovering the k-cluster partition from
     * it.
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

        $samples = $dataset->samples();

        $pi = range(0, $n - 1);
        $lambda = array_fill(0, $n, INF);
        $mu = array_fill(0, $n, 0.0);

        for ($i = 1; $i < $n; ++$i) {
            $pi[$i] = $i;
            $lambda[$i] = INF;

            for ($j = 0; $j < $i; ++$j) {
                $mu[$j] = $this->kernel->compute($samples[$j], $samples[$i]);
            }

            for ($j = 0; $j < $i; ++$j) {
                if ($lambda[$j] >= $mu[$j]) {
                    $mu[$pi[$j]] = min($mu[$pi[$j]], $lambda[$j]);

                    $lambda[$j] = $mu[$j];
                    $pi[$j] = $i;
                } else {
                    $mu[$pi[$j]] = min($mu[$pi[$j]], $mu[$j]);
                }
            }

            for ($j = 0; $j < $i; ++$j) {
                if ($lambda[$j] >= $lambda[$pi[$j]]) {
                    $pi[$j] = $i;
                }
            }
        }

        $order = range(0, $n - 1);

        usort($order, fn ($a, $b) => $lambda[$a] <=> $lambda[$b]);

        $maxMerges = $n - $this->k;

        $parent = range(0, $n - 1);

        $this->losses = [];

        $merges = 0;

        for ($t = 0; $t < $n - 1; ++$t) {
            if ($merges === $maxMerges) {
                break;
            }

            $i = $order[$t];

            $distance = $lambda[$i];

            if (is_nan($distance)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            if ($distance > $this->maxDistance) {
                break;
            }

            $a = $i;

            while ($parent[$a] !== $a) {
                $a = $parent[$a];
            }

            $b = $pi[$i];

            while ($parent[$b] !== $b) {
                $b = $parent[$b];
            }

            if ($a !== $b) {
                $parent[$b] = $a;
            }

            if ($this->logger) {
                $this->logger->info('Epoch: ' . ($merges + 1) . ", Distance: $distance");
            }

            $this->losses[++$merges] = $distance;
        }

        $clusters = $map = [];

        for ($i = 0; $i < $n; ++$i) {
            $root = $i;

            while ($parent[$root] !== $root) {
                $root = $parent[$root];
            }

            if (!isset($map[$root])) {
                $map[$root] = count($clusters);

                $clusters[] = [];
            }

            $clusters[$map[$root]][] = $i;
        }

        $count = count($clusters);

        $this->sizes = array_map('count', $clusters);
        $this->medoids = [];

        foreach ($clusters as $members) {
            $this->medoids[] = $samples[$this->medoid($samples, $members)];
        }

        if ($count !== $this->k) {
            if ($this->logger) {
                $this->logger->warning("Training terminated early, $count clusters were formed,"
                    . " expected {$this->k}.");
            }
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
        $medoids = $this->medoids;

        if (!$medoids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($medoids)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Label a given sample based on its distance from each cluster medoid.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        return argmin($this->medoidDistances($sample));
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
        $medoids = $this->medoids;

        if (!$medoids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($medoids)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the *k* cluster medoids.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        $distances = $proba = [];

        foreach ($this->medoids ?? [] as $medoid) {
            $distances[] = $this->kernel->compute($sample, $medoid) ?: EPSILON;
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
     * Compute the distance from a sample to each cluster medoid.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return list<float>
     */
    protected function medoidDistances(array $sample) : array
    {
        $distances = [];

        foreach ($this->medoids ?? [] as $medoid) {
            $distances[] = $this->kernel->compute($sample, $medoid);
        }

        return $distances;
    }

    /**
     * Return the offset of the medoid of a cluster i.e. the member with the
     * lowest total dissimilarity to the rest of its cluster.
     *
     * @param list<list<string|int|float>> $samples
     * @param list<int> $members
     * @return int
     */
    protected function medoid(array $samples, array $members) : int
    {
        $medoid = $members[0];
        $min = INF;

        foreach ($members as $i) {
            $sum = 0.0;

            foreach ($members as $j) {
                if ($i === $j) {
                    continue;
                }

                $sum += $this->kernel->compute($samples[$i], $samples[$j]);
            }

            if ($sum < $min) {
                $min = $sum;
                $medoid = $i;
            }
        }

        return $medoid;
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
        return 'SLINK (' . Params::stringify($this->params()) . ')';
    }
}
