<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\ITree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

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
class IsolationForest implements Learner, Persistable
{
    /**
     * The number of estimators to train in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The ratio of training samples to train each estimator on.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The amount of contamination (outliers) that is presumed to be in
     * the training set as a percentage.
     *
     * @var float
     */
    protected $contamination;

    /**
     * The average depth of an isolation tree.
     *
     * @var float|null
     */
    protected $pHat;

    /**
     * The trees that make up the forest.
     *
     * @var array
     */
    protected $forest;

    /**
     * The isolation score offset used by the decision function.
     *
     * @var float|null
     */
    protected $offset;

    /**
     * @param int $estimators
     * @param float $ratio
     * @param float $contamination
     * @throws \InvalidArgumentException
     */
    public function __construct(int $estimators = 300, float $ratio = 0.2, float $contamination = 0.1)
    {
        if ($estimators < 1) {
            throw new InvalidArgumentException('The number of estimators'
                . " cannot be less than 1, $estimators given.");
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0.01 and 1, $ratio given.");
        }

        if ($contamination < 0. or $contamination > 0.5) {
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
        return self::DETECTOR;
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
        return $this->offset and $this->forest;
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
        
        $maxDepth = (int) ceil(log(max($n, 2), 2));

        $p = (int) round($this->ratio * $n);

        $this->forest = [];

        for ($epoch = 1; $epoch <= $this->estimators; $epoch++) {
            $tree = new ITree($maxDepth);

            $subset = $dataset->randomize()->head($p);

            $tree->grow($subset);

            $this->forest[] = $tree;
        }

        $this->pHat = $this->c($p);

        $scores = $this->score($dataset);

        $p = 100. - (100. * $this->contamination);

        $shift = Stats::percentile($scores, $p);

        $this->offset = $shift;
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
        if ($this->offset === null) {
            throw new RuntimeException('The learner has not'
                . ' been trained.');
        }

        return array_map([self::class, 'decide'], $this->score($dataset));
    }

    /**
     * Return the isolation scores of each sample in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function score(Dataset $dataset) : array
    {
        if (empty($this->forest)) {
            throw new RuntimeException('The learner has not'
                . ' been trained.');
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

        foreach ($this->forest as $tree) {
            $node = $tree->search($sample);

            $depth += $node ? $node->depth() : self::EPSILON;
        }

        return 2. ** -($depth / $this->estimators / $this->pHat);
    }

    /**
     * Calculate the average path length of an unsuccessful search for n nodes.
     *
     * @param int $n
     * @return float
     */
    protected function c(int $n) : float
    {
        if ($n < 1) {
            return 1.;
        }
        return 2. * (log($n - 1) + M_EULER) - 2. * ($n - 1) / $n;
    }

    /**
     * The decision function.
     *
     * @param float $score
     * @return int
     */
    protected function decide(float $score) : int
    {
        return $score > $this->offset ? 1 : 0;
    }
}
