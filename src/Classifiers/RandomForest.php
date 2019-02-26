<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Functions\Argmax;
use InvalidArgumentException;
use RuntimeException;

/**
 * Random Forest
 *
 * Ensemble classifier that trains Decision Trees (Classification Trees or Extra
 * Trees) on a random subset of the training data. A prediction is made based on
 * the average probability score returned from each tree in the forest.
 *
 * References:
 * [1] L. Breiman. (2001). Random Forests.
 * [2] L. Breiman et al. (2005). Extremely Randomized Trees.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RandomForest implements Learner, Probabilistic, Persistable
{
    protected const AVAILABLE_TREES = [
        ClassificationTree::class,
        ExtraTreeClassifier::class,
    ];

    /**
     * The base estimator.
     *
     * @var \Rubix\ML\Learner
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
     * The number of feature columns in the training set.
     *
     * @var int
     */
    protected $featureCount;

    /**
     * @param \Rubix\ML\Learner|null $base
     * @param int $estimators
     * @param float $ratio
     * @throws \InvalidArgumentException
     */
    public function __construct(?Learner $base = null, int $estimators = 100, float $ratio = 0.1)
    {
        $base = $base ?: new ClassificationTree();

        if (!in_array(get_class($base), self::AVAILABLE_TREES)) {
            throw new InvalidArgumentException('Base estimator is not'
                . ' compatible with this ensemble.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('The number of estimators in the'
                . " ensemble cannot be less than 1, $estimators given.");
        }

        if ($ratio < 0.01 or $ratio > 0.99) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0.01 and 0.99, $ratio given.");
        }

        $this->base = $base;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !empty($this->forest);
    }

    /**
     * Return the feature importances calculated during training keyed by
     * feature column.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function featureImportances() : array
    {
        if (!$this->forest or !$this->featureCount) {
            return [];
        }

        $importances = array_fill(0, $this->featureCount, 0.);

        foreach ($this->forest as $tree) {
            foreach ($tree->featureImportances() as $column => $value) {
                $importances[$column] += $value;
            }
        }

        $k = count($this->forest);

        foreach ($importances as &$importance) {
            $importance /= $k;
        }

        return $importances;
    }

    /**
     * Train a Random Forest by training an ensemble of decision trees on random
     * subsets of the training data.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        $this->classes = $dataset->possibleOutcomes();
        $this->featureCount = $dataset->numColumns();

        $k = (int) round($this->ratio * $dataset->numRows());

        $this->forest = [];

        for ($i = 0; $i < $this->estimators; $i++) {
            $estimator = clone $this->base;

            $subset = $dataset->randomSubsetWithReplacement($k);

            $estimator->train($subset);

            $this->forest[] = $estimator;
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([Argmax::class, 'compute'], $this->proba($dataset));
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->forest)) {
            throw new RuntimeException('The learner has not'
                . ' been trained.');
        }

        $probabilities = array_fill(
            0,
            $dataset->numRows(),
            array_fill_keys($this->classes, 0.)
        );

        foreach ($this->forest as $tree) {
            foreach ($tree->proba($dataset) as $i => $joint) {
                foreach ($joint as $class => $probability) {
                    $probabilities[$i][$class] += $probability;
                }
            }
        }

        $k = count($this->forest);

        foreach ($probabilities as &$joint) {
            foreach ($joint as &$probability) {
                $probability /= $k;
            }
        }

        return $probabilities;
    }
}
