<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use SplQueue;

use function Rubix\ML\argmax;
use function array_fill;
use function array_fill_keys;
use function array_map;
use function array_count_values;
use function array_sum;
use function count;

/**
 * DBSCAN
 *
 * *Density-Based Spatial Clustering of Applications with Noise* is a clustering algorithm
 * able to find non-linearly separable and arbitrarily-shaped clusters given a radius and
 * density constraint. In addition, DBSCAN also has the ability to mark outliers as *noise*
 * and thus can be used as a *quasi* anomaly detector.
 *
 * During training the algorithm is run once on the training set, after which the non-noisy
 * clustered samples are stored in a spatial tree. Unseen samples are then assigned to the
 * cluster that is most common among the samples within *radius* of them during inference, or
 * as noise if no samples are within *radius*.
 *
 * References:
 * [1] M. Ester et al. (1996). A Density-Based Algorithm for Discovering Clusters.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DBSCAN implements Estimator, Learner, Probabilistic, Persistable
{
    use AutotrackRevisions;

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
     * The maximum distance between two points to be considered neighbors. The smaller the value,
     * the tighter the clusters will be.
     *
     * @var float
     */
    protected float $radius;

    /**
     * The minimum number of points to from a dense region or cluster.
     *
     * @var int
     */
    protected int $minDensity;

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
     * @param float $radius
     * @param int $minDensity
     * @param bool $weighted
     * @param Spatial|null $tree
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $radius = 1.0,
        int $minDensity = 5,
        bool $weighted = false,
        ?Spatial $tree = null
    ) {
        if ($radius <= 0.0) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        if ($minDensity <= 0) {
            throw new InvalidArgumentException('Minimum density must be'
                . " greater than 0, $minDensity given.");
        }

        $this->radius = $radius;
        $this->minDensity = $minDensity;
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
     * @return list<\Rubix\ML\DataType>
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

        $labels = range(0, $dataset->numSamples() - 1);

        $dataset = Labeled::quick($dataset->samples(), $labels);

        $this->tree->grow($dataset);

        $cluster = self::START_CLUSTER;

        $predictions = [];

        foreach ($dataset->samples() as $i => $sample) {
            if (isset($predictions[$i])) {
                continue;
            }

            [, $indices] = $this->tree->range($sample, $this->radius);

            if (count($indices) < $this->minDensity) {
                $predictions[$i] = self::NOISE;

                continue;
            }

            $queue = new SplQueue();

            $predictions[$i] = $cluster;

            foreach ($indices as $index) {
                $index = (int) $index;

                if (isset($predictions[$index])) {
                    if ($predictions[$index] === self::NOISE) {
                        $predictions[$index] = $cluster;
                    }

                    continue;
                }

                $predictions[$index] = $cluster;

                $queue->enqueue($index);
            }

            while (!$queue->isEmpty()) {
                $index = $queue->dequeue();

                [, $seeds] = $this->tree->range($dataset->sample($index), $this->radius);

                if (count($seeds) < $this->minDensity) {
                    continue;
                }

                foreach ($seeds as $seed) {
                    $seed = (int) $seed;

                    if (!isset($predictions[$seed])) {
                        $predictions[$seed] = $cluster;

                        $queue->enqueue($seed);
                    } elseif ($predictions[$seed] === self::NOISE) {
                        $predictions[$seed] = $cluster;
                    }
                }
            }

            ++$cluster;
        }

        $this->tree->destroy();

        $samples = $labels = [];

        foreach ($dataset->samples() as $i => $sample) {
            $label = $predictions[$i];

            if ($label === self::NOISE) {
                continue;
            }

            $samples[] = $sample;
            $labels[] = $label;
        }

        if ($samples === []) {
            throw new RuntimeException('No samples remaining to form clusters.');
        }

        $this->clusterCount = $cluster;
        $this->featureCount = $dataset->numFeatures();

        $this->tree->grow(Labeled::quick($samples, $labels));
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
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'DBSCAN (' . Params::stringify($this->params()) . ')';
    }
}
