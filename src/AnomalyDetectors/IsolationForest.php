<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Parallel;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Depth;
use Rubix\ML\Graph\Trees\ITree;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Tasks\Task;
use Rubix\ML\Traits\Multiprocessing;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function array_merge;
use function ceil;

use const Rubix\ML\EPSILON;

/**
 * Isolation Forest
 *
 * An ensemble of Isolation Trees all of which specialize on a unique subset of the training
 * set. Isolation Trees are a type of randomized decision tree that assign anomaly scores
 * based on the depth a sample reaches when traversing the tree. Anomalies are isolated into
 * the shallowest leaf nodes and as such receive the highest *isolation* scores.
 *
 * References:
 * [1] F. T. Liu et al. (2008). Isolation Forest.
 * [2] F. T. Liu et al. (2011). Isolation-based Anomaly Detection.
 * [3] M. Garchery et al. (2018). On the influence of categorical features in ranking anomalies using mixed data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class IsolationForest implements Estimator, Learner, Scoring, Parallel, Persistable
{
    use AutotrackRevisions, Multiprocessing;

    /**
     * The default minimum anomaly score for a sample to be flagged.
     *
     * @var float
     */
    public const DEFAULT_THRESHOLD = 0.5;

    /**
     * The minimum size of each training subset.
     *
     * @var int
     */
    protected const MIN_SUBSAMPLE = 1;

    /**
     * The default sample size of each training subset.
     *
     * @var int
     */
    protected const DEFAULT_SUBSAMPLE = 256;

    /**
     * The number of estimators to train in the ensemble.
     *
     * @var int
     */
    protected int $estimators;

    /**
     * The ratio of training samples to train each estimator on.
     *
     * @var float|null
     */
    protected ?float $ratio = null;

    /**
     * The proportion of outliers that are presumed to be present in the training set.
     *
     * @var float|null
     */
    protected ?float $contamination = null;

    /**
     * The sum of the average depth of all the isolation trees in the ensemble.
     *
     * @var float|null
     */
    protected ?float $delta = null;

    /**
     * The isolation trees that make up the forest.
     *
     * @var ITree[]
     */
    protected array $trees = [
        //
    ];

    /**
     * The isolation score threshold used by the decision function.
     *
     * @var float|null
     */
    protected ?float $threshold = null;

    /**
     * The dimensionality of the training set.
     *
     * @var int|null
     */
    protected ?int $featureCount = null;

    /**
     * Grow an isolation tree on a unique subset of the training set.
     *
     * @internal
     *
     * @param Dataset $subset
     * @param int $maxHeight
     * @return ITree
     */
    public static function growTree(Dataset $subset, int $maxHeight) : ITree
    {
        $tree = new ITree($maxHeight);

        $tree->grow($subset);

        return $tree;
    }

    /**
     * @param int $estimators
     * @param float|null $ratio
     * @param float|null $contamination
     * @throws InvalidArgumentException
     */
    public function __construct(int $estimators = 100, ?float $ratio = null, ?float $contamination = null)
    {
        if ($estimators < 1) {
            throw new InvalidArgumentException('Number of estimators'
                . " must be greater than 0, $estimators given.");
        }

        if (isset($ratio) and ($ratio <= 0.0 or $ratio > 1.0)) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        if (isset($contamination) and ($contamination < 0.0 or $contamination > 0.5)) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->contamination = $contamination;
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::anomalyDetector();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::categorical(),
            DataType::continuous(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'estimators' => $this->estimators,
            'ratio' => $this->ratio,
            'contamination' => $this->contamination,
        ];
    }

    /**
     * Return the parallel processing backend, initializing it with the default if it has
     * not been set yet.
     *
     * @internal
     *
     * @return Backend
     */
    public function backend() : Backend
    {
        return $this->backend ??= new Serial();
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->threshold and $this->trees;
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

        $n = $dataset->numSamples();

        $p = $this->ratio
            ? max(self::MIN_SUBSAMPLE, (int) round($this->ratio * $n))
            : min(self::DEFAULT_SUBSAMPLE, $n);

        $maxHeight = (int) max(1, round(log($p, 2.0)));

        $this->backend()->flush();

        for ($i = 0; $i < $this->estimators; ++$i) {
            $subset = $dataset->randomSubset($p);

            $this->backend()->enqueue(new Task([self::class, 'growTree'], [$subset, $maxHeight]));
        }

        $this->trees = $this->backend()->process();

        $this->delta = $this->estimators * Depth::c($p);

        if (isset($this->contamination)) {
            $scores = $this->scoreChunk($dataset);

            $threshold = Stats::quantile($scores, 1.0 - $this->contamination);
        }

        $this->threshold = $threshold ?? self::DEFAULT_THRESHOLD;

        $this->featureCount = $dataset->numFeatures();
    }

    /**
     * Make predictions from a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->trees) or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $chunkSize = (int) ceil($dataset->numSamples() / $this->backend()->workers());

        $this->backend()->flush();

        foreach ($dataset->batch($chunkSize) as $chunk) {
            $this->backend()->enqueue(new Task([$this, 'predictChunk'], [$chunk]));
        }

        $predictions = [];

        foreach ($this->backend()->process() as $output) {
            /** @var list<int> $output */
            $predictions = array_merge($predictions, $output);
        }

        return $predictions;
    }

    /**
     * Predict a chunk of samples.
     *
     * @internal
     *
     * @param Dataset $chunk
     * @return list<int>
     */
    public function predictChunk(Dataset $chunk) : array
    {
        return array_map([$this, 'predictSample'], $chunk->samples());
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
        return $this->isolationScore($sample) > $this->threshold ? 1 : 0;
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float>
     */
    public function score(Dataset $dataset) : array
    {
        if (empty($this->trees) or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $chunkSize = (int) ceil($dataset->numSamples() / $this->backend()->workers());

        $this->backend()->flush();

        foreach ($dataset->batch($chunkSize) as $chunk) {
            $this->backend()->enqueue(new Task([$this, 'scoreChunk'], [$chunk]));
        }

        $scores = [];

        foreach ($this->backend()->process() as $output) {
            /** @var list<float> $output */
            $scores = array_merge($scores, $output);
        }

        return $scores;
    }

    /**
     * Score a chunk of samples.
     *
     * @internal
     *
     * @param Dataset $chunk
     * @return list<float>
     */
    public function scoreChunk(Dataset $chunk) : array
    {
        return array_map([$this, 'isolationScore'], $chunk->samples());
    }

    /**
     * Return the isolation score of a sample.
     *
     * @param list<string|int|float> $sample
     * @return float
     */
    protected function isolationScore(array $sample) : float
    {
        $depth = 0.0;

        foreach ($this->trees as $tree) {
            $node = $tree->search($sample);

            $depth += $node ? $node->depth() : EPSILON;
        }

        $depth /= $this->delta;

        return 2.0 ** -$depth;
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['backend']);

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
        return 'Isolation Forest (' . Params::stringify($this->params()) . ')';
    }
}
