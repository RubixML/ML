<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use Rubix\ML\Other\Functions\Argmax;
use InvalidArgumentException;
use RuntimeException;
use ReflectionClass;

/**
 * Random Forest
 *
 * Ensemble classifier that trains Decision Trees (Classification Trees or Extra
 * Trees) on a random subset of the training data. A prediction is made based on
 * the probability scores returned from each tree in the forest.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RandomForest implements Estimator, Ensemble, Probabilistic, Persistable
{
    const AVAILABLE_TREES = [
        ClassificationTree::class,
        ExtraTreeClassifier::class,
    ];

    /**
     * The number of trees to train in the ensemble.
     *
     * @var int
     */
    protected $trees;

    /**
     * The ratio of training samples to train each decision tree on.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The maximum depth of a branch before the tree is terminated.
     *
     * @var int
     */
    protected $maxDepth;

    /**
     * The minimum number of samples that each node must contain in order to
     * form a consensus to make a prediction.
     *
     * @var int
     */
    protected $minSamples;

    /**
     * The maximum number of features to consider when determining a split.
     *
     * @var int
     */
    protected $maxFeatures;

    /**
     * A small amount of gini impurity to tolerate when choosing a perfect split.
     *
     * @var float
     */
    protected $tolerance;

    /**
     * The class name of the base classification tree.
     *
     * @var string
     */
    protected $base;

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
     * @param  int  $trees
     * @param  float  $ratio
     * @param  int  $maxDepth
     * @param  int  $minSamples
     * @param  float  $tolerance
     * @param  string  $base
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $trees = 100, float $ratio = 0.1, int $maxDepth = 10,
            int $minSamples = 5, int $maxFeatures = PHP_INT_MAX, float $tolerance = 1e-3,
            string $base = ClassificationTree::class)
    {
        if ($trees < 1) {
            throw new InvalidArgumentException('The number of trees cannot be'
                . ' less than 1.');
        }

        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Sample ratio must be between'
                . ' 0.01 and 1.');
        }

        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have depth less'
                . ' than 1.');
        }

        if ($minSamples < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . ' to make a decision.');
        }

        if ($maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . ' feature to determine a split.');
        }

        if ($tolerance < 0) {
            throw new InvalidArgumentException('Gini tolerance must be 0 or'
                . ' greater.');
        }

        $reflector = new ReflectionClass($base);

        if (!in_array($reflector->getName(), self::AVAILABLE_TREES)) {
            throw new InvalidArgumentException('Base classifier must be a'
                . ' type of classification tree.');
        }

        $this->trees = $trees;
        $this->ratio = $ratio;
        $this->maxDepth = $maxDepth;
        $this->minSamples = $minSamples;
        $this->maxFeatures = $maxFeatures;
        $this->tolerance = $tolerance;
        $this->base = $base;
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

        $n = (int) round($this->ratio * $dataset->numRows());

        $this->forest = [];

        for ($epoch = 0; $epoch < $this->trees; $epoch++) {
            $tree = new $this->base($this->maxDepth, $this->minSamples,
                $this->maxFeatures, $this->tolerance);

            $tree->train($dataset->randomSubsetWithReplacement($n));

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

        $n = count($this->forest);

        $probabilities = array_fill(0, $dataset->numRows(),
            array_fill_keys($this->classes, 0.0));

        foreach ($this->forest as $tree) {
            foreach ($tree->proba($dataset) as $i => $distribution) {
                foreach ($distribution as $class => $probability) {
                    $probabilities[$i][$class] += $probability / $n;
                }
            }
        }

        return $probabilities;
    }
}
