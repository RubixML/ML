<?php

namespace Rubix\Engine;

use MathPHP\Statistics\Average;
use Rubix\Engine\Connectors\Persistable;
use InvalidArgumentException;

class DecisionForest implements Classifier, Regression, Persistable
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
     * The CART trees that make up the forest.
     *
     * @var array
     */
    protected $forest = [
        //
    ];

    /**
     * The output type. i.e. categorical or continuous.
     *
     * @var int
     */
    protected $output;

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
     * @param  \Rubix\Engine\Dataset  $data
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $data) : void
    {
        if (!$data instanceof SupervisedDataset) {
            throw new InvalidArgumentException('This estimator requires a supervised dataset.');
        }

        $this->output = $data->outcomeType();
        $this->forest = [];

        for ($i = 0; $i < $this->trees; $i++) {
            $tree = new CART($this->minSamples, $this->maxDepth);

            $tree->train($data->generateRandomSubset($this->ratio));

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
            return $tree->predict($sample);
        }, $this->forest);

        $outcomes = array_map(function ($prediction) {
            return $prediction->outcome();
        }, $predictions);

        if ($this->output === self::CATEGORICAL) {
            $counts = array_count_values($outcomes);

            $certainty = array_reduce($predictions, function ($carry, $prediction) {
                return $carry += $prediction->meta('certainty');
            }, 0.0) / count($predictions);

            return new Prediction(array_search(max($counts), $counts), [
                'certainty' => $certainty,
            ]);
        } else {
            $variance = array_reduce($predictions, function ($carry, $prediction) {
                return $carry += $prediction->meta('variance');
            }, 0.0) / count($predictions);

            return new Prediction(Average::mean($outcomes), [
                'variance' => $variance,
            ]);
        }
    }
}
