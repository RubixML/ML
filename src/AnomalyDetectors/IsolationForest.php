<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\Trees\ITree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Isolation Forest
 *
 * An Ensemble Anomaly Detector comprised of Isolation Trees each trained on a
 * different subset of the training set. The Isolation Forest works by averaging
 * the isolation score of a sample across a user-specified number of trees.
 *
 * References:
 * [1] F. T. Liu et al. (2008). Isolation Forest.
 * [2] F. T. Liu et al. (2011). Isolation-based Anomaly Detection.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class IsolationForest implements Estimator, Learner, Ranking, Persistable
{
    protected const DEFAULT_SUBSAMPLE = 256;

    public const DEFAULT_THRESHOLD = 0.5;

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
     * The amount of contamination (outliers) that is presumed to be in
     * the training set as a percentage.
     *
     * @var float|null
     */
    protected $contamination;

    /**
     * The average depth of an isolation tree.
     *
     * @var float|null
     */
    protected $delta;

    /**
     * The trees that make up the forest.
     *
     * @var array
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
     * @param int $estimators
     * @param float|null $ratio
     * @param float|null $contamination
     * @throws \InvalidArgumentException
     */
    public function __construct(int $estimators = 100, ?float $ratio = null, ?float $contamination = null)
    {
        if ($estimators < 1) {
            throw new InvalidArgumentException('The number of estimators'
                . " cannot be less than 1, $estimators given.");
        }

        if (isset($ratio) and ($ratio <= 0. or $ratio > 1.)) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0 and 1, $ratio given.");
        }

        if (isset($ratio) and ($contamination < 0. or $contamination > 0.5)) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->contamination = $contamination;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::ANOMALY_DETECTOR;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CATEGORICAL,
            DataType::CONTINUOUS,
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
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $n = $dataset->numRows();

        $k = $this->ratio ? (int) round($this->ratio * $n)
            : min(self::DEFAULT_SUBSAMPLE, $n);

        $maxDepth = (int) ceil(log(max($n, 2), 2));

        $this->trees = [];

        for ($i = 0; $i < $this->estimators; $i++) {
            $tree = new ITree($maxDepth);

            $subset = $dataset->randomize()->head($k);

            $tree->grow($subset);

            $this->trees[] = $tree;
        }

        $this->delta = ITree::c($k);

        if ($this->contamination) {
            $scores = array_map([self::class, 'isolationScore'], $dataset->samples());

            $threshold = Stats::percentile($scores, 100. - (100. * $this->contamination));
        } else {
            $threshold = self::DEFAULT_THRESHOLD;
        }

        $this->threshold = $threshold;
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->threshold === null) {
            throw new RuntimeException('The estimator has not been trained.');
        }

        return array_map([self::class, 'decide'], $this->rank($dataset));
    }

    /**
     * Apply an arbitrary unnormalized scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function rank(Dataset $dataset) : array
    {
        if (empty($this->trees)) {
            throw new RuntimeException('The estimator has not been trained.');
        }
        
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'isolationScore'], $dataset->samples());
    }

    /**
     * Return the isolation score of a sample.
     *
     * @param array $sample
     * @return float
     */
    protected function isolationScore(array $sample) : float
    {
        $depth = 0.;

        foreach ($this->trees as $tree) {
            $node = $tree->search($sample);

            $depth += $node ? $node->depth() : EPSILON;
        }

        $depth /= $this->estimators;

        return 2. ** -($depth / $this->delta);
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
}
