<?php

namespace Rubix\Engine;

use MathPHP\Statistics\Average;
use InvalidArgumentException;

class DecisionForest implements Classifier, Regression
{
    /**
     * The CART trees that make up the forest.
     *
     * @var array
     */
    protected $forest = [
        //
    ];

    /**
     * The number of trees to plant (train) in the ensemble.
     *
     * @var int
     */
    protected $trees;

    /**
     * The ratio of training samples to include in each subset of training data.
     *
     * @var float
     */
    protected $ratio;

    /**
     *  The minimum number of samples that form a consensus to make a prediction.
     *
     * @var int
     */
    protected $minSamples;

    /**
     * The maximum depth of a branch before it is terminated.
     *
     * @var int
     */
    protected $maxDepth;

    /**
     * @param  int  $trees
     * @param  float  $ratio
     * @param  int  $minSize
     * @param  int  $maxHeight
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $trees = 10, float $ratio = 0.1, int $minSamples = 5, int $maxDepth = 1000)
    {
        if ($trees < 1) {
            throw new InvalidArgumentException('The number of trees cannot be less than 1.');
        }

        if ($ratio < 0.1 || $ratio > 1.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value between 0.1 and 1.0.');
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
     * Train an n-tree Decision Forest by generating random subsets of the training
     * data per CART tree.
     *
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return void
     */
    public function train(SupervisedDataset $data) : void
    {
        $this->forest = [];

        foreach (range(1, $this->trees) as $i) {
            $tree = new CART($this->minSamples, $this->maxDepth);

            $tree->train($data->generateRandomSubset($this->ratio));

            $this->forest[] = $tree;
        }
    }

    /**
     * Make a prediction on a given sample.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array
    {
        $outcomes = [];

        foreach ($this->forest as $tree) {
            $outcomes[] = $tree->predict($sample);
        }

        return [
            'outcome' => Average::mode(array_column($outcomes, 'outcome'))[0],
            'certainty' => Average::mean(array_column($outcomes, 'certainty')),
        ];
    }
}
