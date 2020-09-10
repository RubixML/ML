<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Ranking;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Depth;
use Rubix\ML\Graph\Trees\ITree;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Helpers\Verifier;
use Rubix\ML\Other\Traits\ScoresSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;
use Stringable;

use function Rubix\ML\warn_deprecated;
use function count;

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
 * [3] M. Garchery et al. (2018). On the influence of categorical features in
 * ranking anomalies using mixed data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class IsolationForest implements Estimator, Learner, Ranking, Persistable, Stringable
{
    use PredictsSingle, ScoresSingle;

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
    protected $estimators;

    /**
     * The ratio of training samples to train each estimator on.
     *
     * @var float|null
     */
    protected $ratio;

    /**
     * The proportion of outliers that are presumed to be present in the
     * training set.
     *
     * @var float|null
     */
    protected $contamination;

    /**
     * The sum of the average depth of all the isolation trees in the ensemble.
     *
     * @var float|null
     */
    protected $delta;

    /**
     * The isolation trees that make up the forest.
     *
     * @var \Rubix\ML\Graph\Trees\ITree[]
     */
    protected $trees = [
        //
    ];

    /**
     * The isolation score threshold used by the decision function.
     *
     * @var float|null
     */
    protected $threshold;

    /**
     * The dimensionality of the training set.
     *
     * @var int|null
     */
    protected $featureCount;

    /**
     * @param int $estimators
     * @param float|null $ratio
     * @param float|null $contamination
     * @throws \InvalidArgumentException
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
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::anomalyDetector();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        Verifier::check([
            DatasetIsNotEmpty::with($dataset),
            SamplesAreCompatibleWithEstimator::with($dataset, $this),
        ]);

        $n = $dataset->numRows();

        $p = $this->ratio
            ? max(self::MIN_SUBSAMPLE, (int) round($this->ratio * $n))
            : min(self::DEFAULT_SUBSAMPLE, $n);

        $maxHeight = (int) max(1, round(log($p, 2)));

        $this->trees = [];

        while (count($this->trees) < $this->estimators) {
            $tree = new ITree($maxHeight);

            $subset = $dataset->randomSubset($p);

            $tree->grow($subset);

            $this->trees[] = $tree;
        }

        $this->delta = $this->estimators * Depth::c($p);

        if (isset($this->contamination)) {
            $scores = array_map([$this, 'isolationScore'], $dataset->samples());

            $threshold = Stats::quantile($scores, 1.0 - $this->contamination);
        }

        $this->threshold = $threshold ?? self::DEFAULT_THRESHOLD;

        $this->featureCount = $dataset->numColumns();
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return int[]
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([$this, 'decide'], $this->score($dataset));
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return float[]
     */
    public function score(Dataset $dataset) : array
    {
        if (empty($this->trees) or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        return array_map([$this, 'isolationScore'], $dataset->samples());
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @deprecated
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return float[]
     */
    public function rank(Dataset $dataset) : array
    {
        warn_deprecated('Rank() is deprecated, use score() instead.');

        return $this->score($dataset);
    }

    /**
     * Return the isolation score of a sample.
     *
     * @param (string|int|float)[] $sample
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
     * The decision function.
     *
     * @param float $score
     * @return int
     */
    protected function decide(float $score) : int
    {
        return $score > $this->threshold ? 1 : 0;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Isolation Forest (' . Params::stringify($this->params()) . ')';
    }
}
