<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
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
use SplPriorityQueue;

use function Rubix\ML\argmax;
use function array_fill;
use function array_fill_keys;
use function array_map;
use function array_reverse;
use function array_sum;
use function array_count_values;
use function count;
use function fdiv;
use function get_object_vars;
use function is_infinite;
use function max;
use function range;
use function sort;

/**
 * OPTICS
 *
 * *Ordering Points To Identify the Clustering Structure* is a density-based clustering
 * algorithm closely related to DBSCAN. Unlike DBSCAN, which requires a single density
 * threshold, OPTICS builds an ordering of the training set along with a *reachability*
 * distance for each sample. Clusters are then extracted from the resulting reachability
 * plot using the *Xi* (steepness) method, which allows clusters of varying density to be
 * discovered in a single pass. Samples that do not belong to any extracted cluster are
 * flagged as *noise*.
 *
 * During training the algorithm is run once on the training set, after which the non-noisy
 * clustered samples are stored in a spatial tree. Unseen samples are then assigned to the
 * cluster that is most common among the samples within *radius* of them during inference,
 * or as noise if no samples are within *radius*.
 *
 * References:
 * [1] M. Ankerst et al. (1999). OPTICS: Ordering Points To Identify the Clustering Structure.
 * [2] E. Schubert et al. (2018). Improving the Cluster Structure Extracted from OPTICS Plots.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OPTICS implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The starting cluster number.
     *
     * @var int
     */
    public const int START_CLUSTER = 0;

    /**
     * The cluster number assigned to noise samples.
     *
     * @var int
     */
    public const int NOISE = -1;

    /**
     * The maximum distance between two points to be considered neighbors. This is an upper
     * bound on the reachability distances considered during training.
     *
     * @var float
     */
    protected float $radius;

    /**
     * The minimum number of points within *radius* of each other to form a dense region.
     *
     * @var int
     */
    protected int $minDensity;

    /**
     * The minimum steepness on the reachability plot that constitutes a cluster boundary.
     *
     * @var float
     */
    protected float $steepness;

    /**
     * The minimum number of samples contained within an extracted cluster.
     *
     * @var int
     */
    protected int $minClusterSize;

    /**
     * Should we consider the distances of our nearest neighbors when making predictions?
     *
     * @var bool
     */
    protected bool $weighted;

    /**
     * The spatial tree used to run range searches.
     *
     * @var Spatial
     */
    protected Spatial $tree;

    /**
     * The dimensionality of the training set.
     *
     * @var int|null
     */
    protected ?int $featureCount = null;

    /**
     * The number of clusters discovered during training.
     *
     * @var int
     */
    protected int $clusterCount = 0;

    /**
     * The cluster ordered list of sample indices from the last training session.
     *
     * @var list<int>|null
     */
    protected ?array $ordering = null;

    /**
     * The reachability distance of each sample from the last training session.
     *
     * @var list<float>|null
     */
    protected ?array $reachability = null;

    /**
     * The core distance of each sample from the last training session.
     *
     * @var list<float>|null
     */
    protected ?array $coreDistances = null;

    /**
     * @param float $radius
     * @param int $minDensity
     * @param float $steepness
     * @param int|null $minClusterSize
     * @param bool $weighted
     * @param Spatial|null $tree
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $radius = 1.0,
        int $minDensity = 5,
        float $steepness = 0.05,
        ?int $minClusterSize = null,
        bool $weighted = false,
        ?Spatial $tree = null
    ) {
        if ($radius <= 0.0) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        if ($minDensity < 1) {
            throw new InvalidArgumentException('Minimum density must be'
                . " greater than 0, $minDensity given.");
        }

        if ($steepness <= 0.0 or $steepness >= 1.0) {
            throw new InvalidArgumentException('Steepness must be'
                . " between 0 and 1, $steepness given.");
        }

        if ($minClusterSize !== null and $minClusterSize < 1) {
            throw new InvalidArgumentException('Minimum cluster size must be'
                . " greater than 0, $minClusterSize given.");
        }

        $this->radius = $radius;
        $this->minDensity = $minDensity;
        $this->steepness = $steepness;
        $this->minClusterSize = $minClusterSize ?? $minDensity;
        $this->weighted = $weighted;
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
            'radius' => $this->radius,
            'min density' => $this->minDensity,
            'steepness' => $this->steepness,
            'min cluster size' => $this->minClusterSize,
            'weighted' => $this->weighted,
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
        return !$this->tree->bare();
    }

    /**
     * Return the cluster ordered list of sample indices from the last training session.
     *
     * @return list<int>|null
     */
    public function ordering() : ?array
    {
        return $this->ordering;
    }

    /**
     * Return the reachability distance of each sample from the last training session.
     *
     * @return list<float>|null
     */
    public function reachability() : ?array
    {
        return $this->reachability;
    }

    /**
     * Return the core distance of each sample from the last training session.
     *
     * @return list<float>|null
     */
    public function coreDistances() : ?array
    {
        return $this->coreDistances;
    }

    /**
     * Return the base spatial tree instance.
     *
     * @return Spatial
     */
    public function tree() : Spatial
    {
        return $this->tree;
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

        $labels = range(0, $dataset->numSamples() - 1);

        $this->tree->grow(Labeled::quick($dataset->samples(), $labels));

        $predecessor = $this->order($dataset);

        $predictions = $this->extractClusters($predecessor);

        $clusterCount = 0;

        foreach ($predictions as $label) {
            if ($label >= self::START_CLUSTER) {
                $clusterCount = max($clusterCount, $label + 1);
            }
        }

        $samples = $assigned = [];

        foreach ($dataset->samples() as $i => $sample) {
            $label = $predictions[$i];

            if ($label === self::NOISE) {
                continue;
            }

            $samples[] = $sample;
            $assigned[] = $label;
        }

        $this->tree->destroy();

        if ($samples === []) {
            throw new RuntimeException('No samples remaining to form clusters.');
        }

        $this->clusterCount = $clusterCount;
        $this->featureCount = $dataset->numFeatures();

        $this->tree->grow(Labeled::quick($samples, $assigned));

        if ($this->logger) {
            $this->logger->info("Found $clusterCount clusters");

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
        if ($this->tree->bare() or $this->featureCount === null) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        [, $labels, $distances] = $this->tree->range($sample, $this->radius);

        if (empty($labels)) {
            return self::NOISE;
        }

        if ($this->weighted) {
            $weights = array_fill_keys($labels, 0.0);

            foreach ($labels as $i => $label) {
                $weights[$label] += 1.0 / (1.0 + $distances[$i]);
            }
        } else {
            $weights = array_count_values($labels);
        }

        /** @var array<int,float> $weights */
        return argmax($weights);
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
        if ($this->tree->bare() or $this->featureCount === null) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Estimate the joint probabilities of a sample belonging to each cluster.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return array<int,float>
     */
    public function probaSample(array $sample) : array
    {
        $dist = array_fill(self::START_CLUSTER, $this->clusterCount, 0.0);

        $dist[self::NOISE] = 0.0;

        [, $labels, $distances] = $this->tree->range($sample, $this->radius);

        if (empty($labels)) {
            $dist[self::NOISE] = 1.0;

            return $dist;
        }

        if ($this->weighted) {
            $weights = array_fill_keys($labels, 0.0);

            foreach ($labels as $i => $label) {
                $weights[$label] += 1.0 / (1.0 + $distances[$i]);
            }
        } else {
            $weights = array_count_values($labels);
        }

        $total = array_sum($weights);

        foreach ($weights as $cluster => $weight) {
            $dist[$cluster] = (float) $weight / $total;
        }

        return $dist;
    }

    /**
     * Compute the OPTICS cluster ordering and reachability distances of the training set.
     *
     * @internal
     *
     * @param Dataset $dataset
     * @return list<int>
     */
    protected function order(Dataset $dataset) : array
    {
        $n = $dataset->numSamples();

        $samples = $dataset->samples();

        $predecessor = array_fill(0, $n, -1);
        $processed = array_fill(0, $n, false);

        $this->reachability = array_fill(0, $n, INF);
        $this->coreDistances = array_fill(0, $n, INF);
        $this->ordering = [];

        $queue = new SplPriorityQueue();

        $queue->setExtractFlags(SplPriorityQueue::EXTR_DATA);

        for ($i = 0; $i < $n; ++$i) {
            if ($processed[$i]) {
                continue;
            }

            $processed[$i] = true;
            $this->reachability[$i] = INF;

            $this->ordering[] = $i;

            $this->processPoint($queue, $i, $samples, $processed, $predecessor);

            while (!$queue->isEmpty()) {
                $point = (int) $queue->extract();

                if ($processed[$point]) {
                    continue;
                }

                $processed[$point] = true;

                $this->ordering[] = $point;

                $this->processPoint($queue, $point, $samples, $processed, $predecessor);
            }
        }

        return $predecessor;
    }

    /**
     * Process a point in the OPTICS ordering: compute its core distance and update the
     * reachability of its unprocessed neighbors within radius.
     *
     * @internal
     *
     * @param SplPriorityQueue<float,int> $queue
     * @param int $point
     * @param list<list<string|int|float>> $samples
     * @param list<bool> $processed
     * @param array<int,int> $predecessor
     */
    protected function processPoint(
        SplPriorityQueue $queue,
        int $point,
        array $samples,
        array $processed,
        array &$predecessor
    ) : void {
        [, $labels, $distances] = $this->tree->range($samples[$point], $this->radius);

        if (count($labels) < $this->minDensity) {
            $this->coreDistances[$point] = INF;

            return;
        }

        $sortedDistances = $distances;

        sort($sortedDistances);

        $coreDistance = (float) $sortedDistances[$this->minDensity - 1];

        $this->coreDistances[$point] = $coreDistance;

        foreach ($labels as $i => $label) {
            $index = (int) $label;

            if ($processed[$index]) {
                continue;
            }

            $reachability = max($coreDistance, $distances[$i]);

            if ($reachability < $this->reachability[$index]) {
                $predecessor[$index] = $point;

                $queue->insert($index, -$reachability);

                $this->reachability[$index] = $reachability;
            }
        }
    }

    /**
     * Extract a set of cluster labels from the reachability plot using the Xi-steepness method.
     *
     * @internal
     *
     * @param list<int> $predecessor
     * @return list<int>
     */
    protected function extractClusters(array $predecessor) : array
    {
        $n = count($this->ordering);

        $plot = [];

        foreach ($this->ordering as $point) {
            $plot[] = $this->reachability[$point];
        }

        $plot[] = INF;

        $predecessors = [];

        foreach ($this->ordering as $point) {
            $predecessors[] = $predecessor[$point];
        }

        $steepnessComplement = 1.0 - $this->steepness;

        $steepUpward = $steepDownward = $upward = $downward = $steepIndices = [];

        for ($i = 0; $i < $n; ++$i) {
            $ratio = fdiv($plot[$i], $plot[$i + 1]);

            $upward[$i] = $ratio < 1.0;
            $downward[$i] = $ratio > 1.0;
            $steepUpward[$i] = $ratio <= $steepnessComplement;
            $steepDownward[$i] = $ratio >= 1.0 / $steepnessComplement;

            if ($steepUpward[$i] or $steepDownward[$i]) {
                $steepIndices[] = $i;
            }
        }

        $downAreas = $clusters = [];

        $index = 0;
        $mib = 0.0;

        foreach ($steepIndices as $steepIndex) {
            if ($steepIndex < $index) {
                continue;
            }

            $slice = $plot[$index];

            for ($i = $index; $i <= $steepIndex; ++$i) {
                if ($plot[$i] > $slice) {
                    $slice = $plot[$i];
                }
            }

            $mib = max($mib, $slice);

            if ($steepDownward[$steepIndex]) {
                $downAreas = $this->filterSteepDownAreas($downAreas, $mib, $steepnessComplement, $plot);

                $start = $steepIndex;
                $end = $this->extendRegion($steepDownward, $upward, $start);

                $downAreas[] = ['start' => $start, 'end' => $end, 'mib' => 0.0];

                $index = $end + 1;
                $mib = $plot[$index];

                continue;
            }

            $downAreas = $this->filterSteepDownAreas($downAreas, $mib, $steepnessComplement, $plot);

            $upStart = $steepIndex;
            $upEnd = $this->extendRegion($steepUpward, $downward, $upStart);

            $index = $upEnd + 1;
            $mib = $plot[$index];

            $upwardClusters = [];

            foreach ($downAreas as $area) {
                $cStart = $area['start'];
                $cEnd = $upEnd;

                if ($plot[$cEnd + 1] * $steepnessComplement < $area['mib']) {
                    continue;
                }

                $max = $plot[$area['start']];

                if ($max * $steepnessComplement >= $plot[$cEnd + 1]) {
                    while ($plot[$cStart + 1] > $plot[$cEnd + 1] and $cStart < $area['end']) {
                        ++$cStart;
                    }
                } elseif ($plot[$cEnd + 1] * $steepnessComplement >= $max) {
                    while ($cEnd > $upStart and $plot[$cEnd - 1] > $max) {
                        --$cEnd;
                    }
                }

                [$cStart, $cEnd] = $this->correctPredecessor(
                    $plot,
                    $predecessors,
                    $this->ordering,
                    $cStart,
                    $cEnd
                );

                if ($cStart === null) {
                    continue;
                }

                if (($cEnd - $cStart + 1) < $this->minClusterSize) {
                    continue;
                }

                if ($cStart > $area['end'] or $cEnd < $upStart) {
                    continue;
                }

                $upwardClusters[] = [$cStart, $cEnd];
            }

            foreach (array_reverse($upwardClusters) as $cluster) {
                $clusters[] = $cluster;
            }
        }

        $labels = $this->labelClusters($clusters);

        return $labels;
    }

    /**
     * Filter the set of steep down areas using the new maximum in between, dropping those that
     * can no longer contain a valid cluster boundary.
     *
     * @internal
     *
     * @param array<array{start:int,end:int,mib:float}> $downAreas
     * @param float $mib
     * @param float $steepnessComplement
     * @param list<float> $plot
     * @return array<array{start:int,end:int,mib:float}>
     */
    protected function filterSteepDownAreas(array $downAreas, float $mib, float $steepnessComplement, array $plot) : array
    {
        if (is_infinite($mib)) {
            return [];
        }

        $filtered = [];

        foreach ($downAreas as $area) {
            if ($mib <= $plot[$area['start']] * $steepnessComplement) {
                $area['mib'] = max($area['mib'], $mib);

                $filtered[] = $area;
            }
        }

        return $filtered;
    }

    /**
     * Extend a steep region until it is maximal, tolerating runs of non-steep points no longer
     * than the minimum density.
     *
     * @internal
     *
     * @param list<bool> $steepPoints
     * @param list<bool> $xwardPoints
     * @param int $start
     * @return int
     */
    protected function extendRegion(array $steepPoints, array $xwardPoints, int $start) : int
    {
        $n = count($steepPoints);

        $nonXward = 0;
        $index = $start;
        $end = $start;

        while ($index < $n) {
            if ($steepPoints[$index]) {
                $nonXward = 0;
                $end = $index;
            } elseif (!$xwardPoints[$index]) {
                ++$nonXward;

                if ($nonXward > $this->minDensity) {
                    break;
                }
            } else {
                return $end;
            }

            ++$index;
        }

        return $end;
    }

    /**
     * Correct the boundaries of a candidate cluster according to its predecessors.
     *
     * @internal
     *
     * @param list<float> $plot
     * @param list<int> $predecessors
     * @param list<int> $ordering
     * @param int $start
     * @param int $end
     * @return array{int|null,int|null}
     */
    protected function correctPredecessor(
        array $plot,
        array $predecessors,
        array $ordering,
        int $start,
        int $end
    ) : array {
        while ($start < $end) {
            if ($plot[$start] > $plot[$end]) {
                return [$start, $end];
            }

            $predecessor = $predecessors[$end];

            for ($i = $start; $i < $end; ++$i) {
                if ($predecessor === $ordering[$i]) {
                    return [$start, $end];
                }
            }

            --$end;
        }

        return [null, null];
    }

    /**
     * Assign a cluster label to each sample given a list of ordered cluster spans.
     *
     * @internal
     *
     * @param list<array{int,int}> $clusters
     * @return list<int>
     */
    protected function labelClusters(array $clusters) : array
    {
        $n = count($this->ordering);

        $labels = array_fill(0, $n, self::NOISE);

        $label = self::START_CLUSTER;

        foreach ($clusters as [$start, $end]) {
            $occupied = false;

            for ($i = $start; $i <= $end; ++$i) {
                if ($labels[$i] !== self::NOISE) {
                    $occupied = true;

                    break;
                }
            }

            if ($occupied) {
                continue;
            }

            for ($i = $start; $i <= $end; ++$i) {
                $labels[$i] = $label;
            }

            ++$label;
        }

        $result = array_fill(0, $n, self::NOISE);

        foreach ($this->ordering as $i => $point) {
            $result[$point] = $labels[$i];
        }

        return $result;
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset(
            $properties['ordering'],
            $properties['reachability'],
            $properties['coreDistances'],
            $properties['logger'],
        );

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
        return 'OPTICS (' . Params::stringify($this->params()) . ')';
    }
}
