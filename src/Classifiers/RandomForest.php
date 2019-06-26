<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Deferred;
use Rubix\ML\Parallel;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
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
class RandomForest implements Estimator, Learner, Probabilistic, Parallel, Persistable
{
    use Multiprocessing;

    /**
     * The base estimator.
     *
     * @var \Rubix\ML\Classifiers\ClassificationTree
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
     * The decision trees that make up the forest.
     *
     * @var array
     */
    protected $trees = [
        //
    ];

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The number of feature columns in the training set.
     *
     * @var int|null
     */
    protected $featureCount;

    /**
     * @param \Rubix\ML\Classifiers\ClassificationTree|null $base
     * @param int $estimators
     * @param float $ratio
     * @throws \InvalidArgumentException
     */
    public function __construct(?ClassificationTree $base = null, int $estimators = 100, float $ratio = 0.1)
    {
        if ($estimators < 1) {
            throw new InvalidArgumentException('The number of estimators'
                . " in the ensemble cannot be less than 1, $estimators"
                . ' given.');
        }

        if ($ratio <= 0. or $ratio >= 1.) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0 and 1, $ratio given.");
        }

        $this->base = $base ?? new ClassificationTree();
        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->backend = new Serial();
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
        return !empty($this->trees);
    }

    /**
     * Train the learner with a dataset.
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

        DatasetIsCompatibleWithEstimator::check($dataset, $this);
        
        $k = (int) round($this->ratio * $dataset->numRows());

        for ($i = 0; $i < $this->estimators; $i++) {
            $estimator = clone $this->base;

            $subset = $dataset->randomSubsetWithReplacement($k);

            $this->backend->enqueue(new Deferred(
                [self::class, 'trainer'],
                [$estimator, $subset]
            ));
        }

        $this->trees = $this->backend->process();

        $this->classes = $dataset->possibleOutcomes();
        $this->featureCount = $dataset->numColumns();
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map('Rubix\ML\argmax', $this->proba($dataset));
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
        if (empty($this->trees)) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        $this->backend->flush();

        foreach ($this->trees as $estimator) {
            $this->backend->enqueue(new Deferred(
                [self::class, 'predictor'],
                [$estimator, $dataset]
            ));
        }

        $aggregate = $this->backend->process();

        $probabilities = array_fill(
            0,
            $dataset->numRows(),
            array_fill_keys($this->classes, 0.)
        );

        foreach ($aggregate as $proba) {
            foreach ($proba as $i => $joint) {
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

    /**
     * Return the feature importances calculated during training keyed by
     * feature column.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function featureImportances() : array
    {
        if (!$this->trees or !$this->featureCount) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        $importances = array_fill(0, $this->featureCount, 0.);

        foreach ($this->trees as $tree) {
            foreach ($tree->featureImportances() as $column => $value) {
                $importances[$column] += $value;
            }
        }

        foreach ($importances as &$importance) {
            $importance /= $this->estimators;
        }

        return $importances;
    }

    /**
     * Train an estimator using a dataset and return it.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Learner
     */
    public static function trainer(Learner $estimator, Dataset $dataset) : Learner
    {
        $estimator->train($dataset);

        return $estimator;
    }

    /**
     * Return the probabilities from a decision tree.
     *
     * @param \Rubix\ML\Probabilistic $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array
     */
    public static function predictor(Probabilistic $estimator, Dataset $dataset) : array
    {
        return $estimator->proba($dataset);
    }
}
