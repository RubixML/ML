<?php

namespace Rubix\Engine;

use MathPHP\Statistics\Average;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Persisters\Persistable;
use InvalidArgumentException;

class RandomForest implements Estimator, Classifier, Regression, Persistable
{
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
     * The output type. i.e. categorical or continuous.
     *
     * @var int
     */
    protected $output;

    /**
     * The CART trees that make up the forest.
     *
     * @var array
     */
    protected $forest = [
        //
    ];

    /**
     * @param  int  $trees
     * @param  float  $ratio
     * @param  int  $minSize
     * @param  int  $maxHeight
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $trees = 10, float $ratio = 0.1, int $minSamples = 5, int $maxDepth = 10)
    {
        if ($trees < 1) {
            throw new InvalidArgumentException('The number of trees cannot be less than 1.');
        }

        if ($ratio < 0.01 || $ratio > 1.0) {
            throw new InvalidArgumentException('Sample ratio must be a float value between 0.01 and 1.0.');
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
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        $this->output = $dataset->outcomeType();
        $this->forest = [];

        for ($i = 0; $i < $this->trees; $i++) {
            $tree = new CART($this->minSamples, $this->maxDepth);

            $tree->train($dataset->generateRandomSubsetWithReplacement($this->ratio));

            $this->forest[$i] = $tree;
        }
    }

    /**
     * Make a prediction on a given sample.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $predictions = array_map(function ($tree) use ($sample) {
            return $tree->predict($sample)->outcome();
        }, $this->forest);

        if ($this->output === self::CATEGORICAL) {
            $counts = array_count_values($predictions);

            $outcome = array_search(max($counts), $counts);
        } else {
            $outcome = Average::mean($predictions);
        }

        return new Prediction($outcome);
    }
}
