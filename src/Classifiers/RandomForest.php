<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Supervised;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use MathPHP\Statistics\Average;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

class RandomForest implements Supervised, Multiclass, Probabilistic, Persistable
{
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
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $trees = 50, float $ratio = 0.1, int $maxDepth = 10,
                                int $minSamples = 5)
    {
        if ($trees < 1) {
            throw new InvalidArgumentException('The number of trees cannot be'
                . ' less than 1.');
        }

        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Sample ratio must be a float'
                . ' value between 0.01 and 1.0.');
        }

        $this->trees = $trees;
        $this->ratio = $ratio;
        $this->minSamples = $minSamples;
        $this->maxDepth = $maxDepth;
    }

    /**
     * Train a Random Forest by training an ensemble of decision trees on random
     * subsets of the training data.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return void
     */
    public function train(Labeled $dataset) : void
    {
        $this->classes = $dataset->possibleOutcomes();

        $n = $this->ratio * count($dataset);

        $this->forest = [];

        for ($i = 0; $i < $this->trees; $i++) {
            $tree = new DecisionTree($this->maxDepth, $this->minSamples);

            $tree->train($dataset->randomSubset($n));

            $this->forest[] = $tree;
        }
    }

    /**
     * Make a prediction based on the class probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($this->proba($samples) as $probabilities) {
            $best = ['probability' => -INF, 'outcome' => null];

            foreach ($probabilities as $class => $probability) {
                if ($probability > $best['probability']) {
                    $best['probability'] = $probability;
                    $best['outcome'] = $class;
                }
            }

            $predictions[] = $best['outcome'];
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function proba(Dataset $samples) : array
    {
        $n = count($this->forest) + self::EPSILON;

        $probabilities = array_fill(0, $samples->numRows(),
            array_fill_keys($this->classes, 0.0));

        foreach ($this->forest as $tree) {
            foreach ($tree->proba($samples) as $i => $results) {
                foreach ($results as $class => $probability) {
                    $probabilities[$i][$class] += $probability / $n;
                }
            }
        }

        return $probabilities;
    }
}
