<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
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
class IsolationForest implements Estimator, Ensemble, Probabilistic, Persistable
{
    const AVAILABLE_ESTIMATORS = [
        IsolationTree::class,
    ];

    /**
     * The base isolation tree to be used in the ensemble.
     * 
     * @var \Rubix\ML\Estimator
     */
    protected $base;

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
     * The trees that make up the forest.
     *
     * @var array
     */
    protected $forest = [
        //
    ];

    /**
     * @param  \Rubix\ML\Estimator|null  $base
     * @param  int  $estimators
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(?Estimator $base = null, int $estimators = 100, float $ratio = 0.1)
    {
        if (is_null($base)) {
            $base = new IsolationTree();
        }

        if (!in_array(get_class($base), self::AVAILABLE_ESTIMATORS)) {
            throw new InvalidArgumentException('Base estimator is not'
                . ' compatible with this ensemble.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException("The number of estimators in the"
                . " ensemble cannot be less than 1, $estimators given.");
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException("Sample ratio must be between"
                . " 0.01 and 1, $ratio given.");
        }

        $this->base = $base;
        $this->estimators = $estimators;
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
     * Return the ensemble of estimators.
     *
     * @return array
     */
    public function estimators() : array
    {
        return $this->forest;
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
        $p = (int) round($this->ratio * $dataset->numRows());

        $this->forest = [];

        for ($epoch = 0; $epoch < $this->estimators; $epoch++) {
            $tree = clone $this->base;

            $subset = $dataset->randomize()->head($p);

            $tree->train($subset);

            $this->forest[] = $tree;
        }
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($this->proba($dataset) as $probability) {
            $predictions[] = $probability > 0.5 ? 1 : 0;
        }

        return $predictions;
    }

    /**
     * Output a probability of being an anomaly.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->forest)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $k = count($this->forest);

        $probabilities = array_fill(0, $dataset->numRows(), 0.);

        foreach ($this->forest as $tree) {
            foreach ($tree->proba($dataset) as $i => $proba) {
                $probabilities[$i] += $proba;
            }
        }

        foreach ($probabilities as &$proba) {
            $proba /= $k;
        }

        return $probabilities;
    }
}
