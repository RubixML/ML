<?php

namespace Rubix\Engine\Classifiers;

use Rubix\Engine\Supervised;
use Rubix\Engine\Persistable;
use MathPHP\Statistics\Average;
use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Labeled;
use InvalidArgumentException;

class RandomForest implements Supervised, Probabilistic, Classifier, Persistable
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
     * The minimum number of samples that each node must contain in order to
     * form a consensus to make a prediction.
     *
     * @var int
     */
    protected $minSamples;

    /**
     * The maximum depth of a branch before the tree is terminated.
     *
     * @var int
     */
    protected $maxDepth;

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
     * @param  int  $minSamples
     * @param  int  $maxDepth
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $trees = 50, float $ratio = 0.1, int $minSamples = 5,
                                int $maxDepth = 10)
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
     * The number of trees in the forest.
     *
     * @return int
     */
    public function trees() : int
    {
        return count($this->forest);
    }

    /**
     * Train a Random Forest by training an ensemble of decision trees on random
     * subsets of the training data.
     *
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @return void
     */
    public function train(Labeled $dataset) : void
    {
        $this->classes = $dataset->possibleOutcomes();

        $n = $this->ratio * count($dataset);

        $this->forest = [];

        for ($i = 0; $i < $this->trees; $i++) {
            $tree = new DecisionTree($this->minSamples, $this->maxDepth);

            $tree->train($dataset->randomSubset($n));

            $this->forest[] = $tree;
        }
    }

    /**
     * Make a prediction based on the class probabilities.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
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
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function proba(Dataset $samples) : array
    {
        $probabilities = array_fill(0, $samples->numRows(),
            array_fill_keys($this->classes, 0.0));

        $n = $this->trees();

        foreach ($this->forest as $tree) {
            foreach ($tree->proba($samples) as $i => $results) {
                foreach ($results as $class => $probability) {
                    $probabilities[$i][$class] += $probability;
                }
            }
        }

        for ($i = 0; $i < count($probabilities); $i++) {
            foreach ($probabilities[$i] as &$probability) {
                $probability /= $n;
            }
        }

        return $probabilities;
    }
}
