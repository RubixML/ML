<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Ensemble;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\ITree;
use Rubix\ML\Other\Helpers\Stats;
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
class IsolationForest implements Learner, Ensemble, Persistable
{
    /**
     * The number of estimators to train in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The amount of contamination (outliers) that is presumed to be in
     * the training set as a percentage.
     *
     * @var float
     */
    protected $contamination;

    /**
     * The ratio of training samples to train each estimator on.
     *
     * @var float
     */
    protected $ratio;

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
    protected $forest = [
        //
    ];

    /**
     * The isolation score offset used by the decision function.
     * 
     * @var float|null
     */
    protected $offset;

    /**
     * @param  int  $estimators
     * @param  float  $contamination
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $estimators = 300, float $contamination = 0.1, float $ratio = 0.2)
    {
        if ($estimators < 1) {
            throw new InvalidArgumentException("The number of estimators in the"
                . " ensemble cannot be less than 1, $estimators given.");
        }

        if ($contamination < 0.) {
            throw new InvalidArgumentException("Contamination cannot be less"
                . " than 0, $contamination given.");
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException("Subsample ratio must be between"
                . " 0.01 and 1, $ratio given.");
        }

        $this->estimators = $estimators;
        $this->contamination = $contamination;
        $this->ratio = $ratio;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::DETECTOR;
    }

    /**
     * Train a Random Forest by training an ensemble of decision trees on random
     * subsets of the training data.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $n = $dataset->numRows();
        
        $maxDepth = (int) ceil(log(max($n, 2), 2));

        $p = (int) round($this->ratio * $n);

        $this->pHat = $this->c($p);

        $this->forest = [];

        for ($epoch = 0; $epoch < $this->estimators; $epoch++) {
            $tree = new ITree($maxDepth);

            $subset = $dataset->randomize()->head($p);

            $tree->grow($subset);

            $this->forest[] = $tree;
        }

        $scores = [];

        foreach ($dataset as $sample) {
            $scores[] = $this->isolationScore($sample);
        }

        $shift = Stats::percentile($scores, 100. * $this->contamination);

        $this->offset = $shift;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (is_null($this->offset)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset as $sample) {
            $score = $this->isolationScore($sample);

            $score -= $this->offset;

            $predictions[] = $score < 0. ? 1 : 0;
        }

        return $predictions;
    }

    /**
     * Return the isolation score of a sample.
     * 
     * @param  array  $sample
     * @return float
     */
    protected function isolationScore(array $sample) : float
    {
        $depths = [];

        foreach ($this->forest as $tree) {
            $node = $tree->search($sample);

            $depths[] = isset($node) ? $node->depth() : self::EPSILON;
        }

        $mean = Stats::mean($depths);

        $score = 2. ** -($mean / $this->pHat);

        return -$score;
    }

    /**
     * Calculate the average path length of an unsuccessful search for n nodes.
     *
     * @param  int  $n
     * @return float
     */
    protected function c(int $n) : float
    {
        if ($n <= 1) {
            return 1.;
        }
        
        return 2. * (log($n - 1) + M_EULER) - 2. * ($n - 1) / $n;
    }
}
