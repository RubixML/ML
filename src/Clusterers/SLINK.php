<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function count;
use function min;
use function range;
use function array_fill;
use function array_map;
use function get_object_vars;
use function is_nan;
use function usort;

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
 * tend to be elongated and cannot be characterized by a mean vector. A sample
 * is therefore assigned to the cluster that contains its *nearest member* —
 * the single-link extension rule — which guarantees that the partition
 * recovered during training is reproduced exactly. The training samples and
 * their cluster labels are stored only in a *spatial tree* that is grown at the
 * end of training, so prediction is a nearest neighbor query rather than a
 * scan of every cluster. Probabilities are estimated from the distance to the
 * nearest member of each cluster, which is recovered with the tree as well.
 *
 * The distance kernel and the data types supported by the estimator are those
 * of the tree. A `BallTree` (Euclidean by default) pairs with any *subadditive*
 * kernel, while a `KDTree` requires a *monotonic* one.
 *
 * References:
 * [1] R. Sibson. (1973). SLINK: An Optimally Efficient Algorithm for the
 * Single-Link Cluster Method.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SLINK implements Estimator, Learner, Verbose, Persistable
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
     * The spatial tree used to store the training samples and their labels and
     * to search for nearest neighbors.
     *
     * @var Spatial
     */
    protected Spatial $tree;

    /**
     * The number of features in the training set.
     *
     * @var int|null
     */
    protected ?int $featureCount = null;

    /**
     * The number of samples in the training set.
     *
     * @var int
     */
    protected int $numSamples = 0;

    /**
     * The number of clusters formed during the last training session.
     *
     * @var int
     */
    protected int $clusterCount = 0;

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
     * @param Spatial|null $tree
     * @throws InvalidArgumentException
     */
    public function __construct(int $k, float $maxDistance = INF, ?Spatial $tree = null)
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
        $this->tree = $tree ?? new BallTree();
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
        return $this->tree->kernel()->compatibility();
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
            'tree' => $this->tree,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->tree->bare() and $this->featureCount !== null;
    }

    /**
     * Return the spatial tree that stores the training samples and their
     * cluster labels.
     *
     * @return Spatial
     */
    public function tree() : Spatial
    {
        return $this->tree;
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

        $kernel = $this->tree->kernel();

        $samples = $dataset->samples();

        $pi = range(0, $n - 1);
        $lambda = array_fill(0, $n, INF);
        $mu = array_fill(0, $n, 0.0);

        for ($i = 1; $i < $n; ++$i) {
            $pi[$i] = $i;
            $lambda[$i] = INF;

            for ($j = 0; $j < $i; ++$j) {
                $mu[$j] = $kernel->compute($samples[$j], $samples[$i]);
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

        $labels = array_fill(0, $n, 0);
        $sizes = [];

        foreach ($clusters as $label => $members) {
            foreach ($members as $offset) {
                $labels[$offset] = $label;
            }

            $sizes[$label] = count($members);
        }

        $this->sizes = $sizes;
        $this->clusterCount = $count;
        $this->numSamples = $n;
        $this->featureCount = $dataset->numFeatures();

        $this->tree->grow(Labeled::quick($samples, $labels));

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
        if (!$this->trained() or $this->featureCount === null) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Label a given sample with the cluster of its nearest training sample.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        [, $labels] = $this->tree->nearest($sample, 1);

        return (int) $labels[0];
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
