<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Parallel;
use Rubix\ML\Deferred;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\argmax;
use function Rubix\ML\array_transpose;
use function get_class;

/**
 * Random Forest
 *
 * An ensemble classifier that trains Decision Trees (Classification Trees or Extra Trees)
 * on a random subset (*bootstrap* set) of the training data. A prediction is made based
 * on the probability scores returned from each tree in the forest which are then averaged
 * and weighted equally.
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
    use Multiprocessing, PredictsSingle, ProbaSingle;

    public const COMPATIBLE_LEARNERS = [
        ClassificationTree::class,
        ExtraTreeClassifier::class,
    ];

    /**
     * The base learner.
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
     * The decision trees that make up the forest.
     *
     * @var array|null
     */
    protected $trees;

    /**
     * The possible class outcomes.
     *
     * @var array|null
     */
    protected $classes;

    /**
     * The number of feature columns in the training set.
     *
     * @var int|null
     */
    protected $featureCount;

    /**
     * @param \Rubix\ML\Learner|null $base
     * @param int $estimators
     * @param float $ratio
     * @throws \InvalidArgumentException
     */
    public function __construct(?Learner $base = null, int $estimators = 100, float $ratio = 0.2)
    {
        if ($base and !in_array(get_class($base), self::COMPATIBLE_LEARNERS)) {
            throw new InvalidArgumentException('Base learner is not'
                . ' compatible with ensemble.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('The number of estimators'
                . " in the ensemble cannot be less than 1, $estimators"
                . ' given.');
        }

        if ($ratio <= 0. or $ratio > 1.5) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0 and 1.5, $ratio given.");
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
            throw new InvalidArgumentException('Learner requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->classes = $dataset->possibleOutcomes();
        $this->featureCount = $dataset->numColumns();
        
        $k = (int) round($this->ratio * $dataset->numRows());

        $this->backend->flush();

        for ($i = 0; $i < $this->estimators; ++$i) {
            $estimator = clone $this->base;

            $subset = $dataset->randomSubsetWithReplacement($k);

            $this->backend->enqueue(new Deferred(
                [self::class, '_train'],
                [$estimator, $subset]
            ));
        }

        $this->trees = $this->backend->process();
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trees) {
            throw new RuntimeException('The estimator has not been trained.');
        }

        $this->backend->flush();

        foreach ($this->trees as $estimator) {
            $this->backend->enqueue(new Deferred(
                [self::class, '_predict'],
                [$estimator, $dataset]
            ));
        }

        $aggregate = array_transpose($this->backend->process());

        $predictions = [];

        foreach ($aggregate as $votes) {
            $predictions[] = argmax(array_count_values($votes));
        }

        return $predictions;
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->trees or !$this->classes) {
            throw new RuntimeException('The estimator has not been trained.');
        }

        $probabilities = array_fill(
            0,
            $dataset->numRows(),
            array_fill_keys($this->classes, 0.)
        );

        $this->backend->flush();

        foreach ($this->trees as $estimator) {
            $this->backend->enqueue(new Deferred(
                [self::class, '_proba'],
                [$estimator, $dataset]
            ));
        }

        $aggregate = $this->backend->process();

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
     * Return the normalized feature importances i.e. the proportion that each
     * feature contributes to the overall model, indexed by feature column.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function featureImportances() : array
    {
        if (!$this->trees or !$this->featureCount) {
            throw new RuntimeException('The estimator has not been trained.');
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
     * Train an estimator using a supplied dataset and return it.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Rubix\ML\Learner
     */
    public static function _train(Learner $estimator, Dataset $dataset) : Learner
    {
        $estimator->train($dataset);

        return $estimator;
    }

    /**
     * Return the predictions from a decision tree.
     *
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return mixed[]
     */
    public static function _predict(Estimator $estimator, Dataset $dataset) : array
    {
        return $estimator->predict($dataset);
    }

    /**
     * Return the probabilities of each class outcome from a decision tree.
     *
     * @param \Rubix\ML\Probabilistic $estimator
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    public static function _proba(Probabilistic $estimator, Dataset $dataset) : array
    {
        return $estimator->proba($dataset);
    }
}
