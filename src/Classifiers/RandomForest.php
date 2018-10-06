<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Functions\Argmax;
use InvalidArgumentException;
use RuntimeException;

/**
 * Random Forest
 *
 * Ensemble classifier that trains Decision Trees (Classification Trees or Extra
 * Trees) on a random subset of the training data. A prediction is made based on
 * the probability scores returned from each tree in the forest.
 *
 * References:
 * [1] L. Breiman. (2001). Random Forests.
 * [2] L. Breiman et al. (2005). Extremely Randomized Trees.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RandomForest implements Estimator, Ensemble, Probabilistic, Persistable
{
    const AVAILABLE_ESTIMATORS = [
        ClassificationTree::class,
        ExtraTreeClassifier::class,
    ];

    /**
     * The base estimator.
     *
     * @var \Rubix\ML\Estimator
     */
    protected $base;

    /**
     * The number of trees to train in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The ratio of training samples to train each decision tree on.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The decision trees that make up the forest.
     *
     * @var array
     */
    protected $forest = [
        //
    ];

    /**
     * @param  \Rubix\ML\Estimator  $base
     * @param  int  $estimators
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(Estimator $base = null, int $estimators = 100, float $ratio = 0.1)
    {
        if (is_null($base)) {
            $base = new ClassificationTree();
        }

        if (!in_array(get_class($base), self::AVAILABLE_ESTIMATORS)) {
            throw new InvalidArgumentException('Base estimator is not'
                . ' compatible with random forest.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('The number of estimators in the'
                . ' ensemble cannot be less than 1.');
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException('Sample ratio must be between'
                . ' 0.01 and 1.');
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
        return self::CLASSIFIER;
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
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $this->classes = $dataset->possibleOutcomes();

        $k = (int) round($this->ratio * $dataset->numRows());

        $this->forest = [];

        for ($epoch = 0; $epoch < $this->estimators; $epoch++) {
            $tree = clone $this->base;

            $subset = $dataset->randomSubsetWithReplacement($k);

            $tree->train($subset);

            $this->forest[] = $tree;
        }
    }

    /**
     * Make a prediction based on the class probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($this->proba($dataset) as $probabilities) {
            $predictions[] = Argmax::compute($probabilities);
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
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

        $probabilities = array_fill(0, $dataset->numRows(),
            array_fill_keys($this->classes, 0.));

        foreach ($this->forest as $tree) {
            foreach ($tree->proba($dataset) as $i => $joint) {
                foreach ($joint as $class => $probability) {
                    $probabilities[$i][$class] += $probability;
                }
            }
        }

        foreach ($probabilities as &$joint) {
            foreach ($joint as &$probability) {
                $probability /= $this->estimators;
            }
        }

        return $probabilities;
    }
}
