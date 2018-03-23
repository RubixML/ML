<?php

namespace Rubix\Engine;

use Rubix\Engine\Math\Stats;
use Rubix\Engine\Math\Random;
use InvalidArgumentException;

class DecisionForest implements Estimator
{
    /**
     * The decision trees that make up the forest.
     *
     * @var array
     */
    protected $forest = [
        //
    ];

    /**
     * The number of trees to train.
     *
     * @var int
     */
    protected $trees;

    /**
     * The ratio of samples to include in each subset of data.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The minimum number of samples that a node needs to make a prediction.
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
    public function __construct(int $trees = 100, float $ratio = 0.1, int $minSamples = 5, int $maxDepth = 1000)
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
     * @return array
     */
    public function forest() : array
    {
        return $this->forest;
    }

    /**
     * Train the Decision Forest on a labeled data set.
     *
     * @param  array  $data
     * @return self
     */
    public function train(array $data) : void
    {
        $this->forest = [];

        foreach (range(1, $this->trees) as $i) {
            $subset = $this->generateRandomSubset($data, $this->ratio);

            $tree = new CART($this->minSamples, $this->maxDepth);

            $tree->train($subset);

            $this->forest[] = $tree;
        }
    }

    /**
     * Calculate the accuracy of the Decision Forest.
     *
     * @return float
     */
    public function test(array $data) : float
    {
        $score = 0;

        foreach ($data as $i => $sample) {
            $actual = array_pop($sample);

            $outcome = $this->predict($sample)['outcome'];

            if ($outcome === $actual) {
                $score++;
            }
        }

        return (float) ($score / count($data));
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
            'outcome' => Stats::mode(array_column($outcomes, 'outcome')),
            'certainty' => Stats::mean(array_column($outcomes, 'certainty')),
        ];
    }

    /**
     * Generate a random subset with replacement of the data set. O(N)
     *
     * @param  array  $data
     * @param  float  $ratio
     * @return array
     */
    protected function generateRandomSubset(array $data, float $ratio) : array
    {
        $n = $ratio * count($data);
        $subset = [];

        foreach (range(1, $n) as $i) {
            $subset[] = Random::item($data);
        }

        return $subset;
    }
}
