<?php

namespace Rubix\ML\AnomalyDetection;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use InvalidArgumentException;

class IsolationForest implements Detector, Probabilistic, Persistable
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
     * The threshold isolation score. Score is a value between 0 and 1 where
     * 0.5 is nominal, 1 is certain to be an outlier, and 0 is an extremely
     * dense region.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The isolation trees that make up the forest.
     *
     * @var array
     */
    protected $forest = [
        //
    ];

    /**
     * @param  int  $trees
     * @param  float  $ratio
     * @param  float  $threshold
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $trees = 100, float $ratio = 0.1, float $threshold = 0.6)
    {
        if ($trees < 1) {
            throw new InvalidArgumentException('The number of trees cannot be'
                . ' less than 1.');
        }

        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Sample ratio must be a float'
                . ' value between 0.01 and 1.0.');
        }

        if ($threshold < 0 or $threshold > 1) {
            throw new InvalidArgumentException('Threshold isolation score must'
                . ' be between 0 and 1.');
        }

        $this->trees = $trees;
        $this->ratio = $ratio;
        $this->threshold = $threshold;
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
        $n = $this->ratio * $dataset->numRows();

        $maxDepth = ceil(log($n, 2));

        $this->forest = [];

        for ($i = 0; $i < $this->trees; $i++) {
            $tree = new IsolationTree($maxDepth, $this->threshold);

            $tree->train($dataset->randomSubset($n));

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
            $predictions[] = $probability > $this->threshold ? 1 : 0;
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $n = count($this->forest) + self::EPSILON;

        $probabilities = [];

        foreach ($dataset as $sample) {
            $probability = 0.0;

            foreach ($this->forest as $tree) {
                $probability += $tree->search($sample)->get('probability');
            }

            $probabilities[] = $probability / $n;
        }

        return $probabilities;
    }
}
