<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use Rubix\ML\Metrics\Validation\Validation;
use InvalidArgumentException;

class KFold implements Validator
{
    /**
     * The metric used to score the predictions.
     *
     * @var \Rubix\ML\Metrics\Validation
     */
    protected $metric;

    /**
     * The number of times to split the dataset and therefore the number of
     * unique testing sets that will be generated.
     *
     * @var int
     */
    protected $folds;

    /**
     * @param  \Rubix\ML\Metrics\Validation\Validation  $metric
     * @param  int  $folds
     * @return void
     */
    public function __construct(Validation $metric, int $folds = 10)
    {
        $this->metric = $metric;
        $this->folds = $folds;
    }

    /**
     * Run k training rounds where k is the number of folds. For each round use
     * one fold for testing and the rest to train the model. Return the average
     * score for each training round.
     *
     * @param  \Rubix\ML\Estimator\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return float
     */
    public function score(Estimator $estimator, Labeled $dataset) : float
    {
        $dataset->randomize();

        if ($estimator instanceof Classifier or $estimator instanceof Clusterer) {
            $folds = $dataset->stratifiedFold($this->folds);
        } else {
            $folds = $dataset->fold($this->folds);
        }

        $scores = [];

        for ($i = 0; $i < count($folds); $i++) {
            $training = [];

            for ($j = 0; $j < count($folds); $j++) {
                if ($i === $j) {
                    $testing = clone $folds[$j];
                } else {
                    $training[] = clone $folds[$j];
                }
            }

            $estimator->train(Labeled::combine($training));

            $scores[] = $this->metric->score($estimator, $testing);
        }

        return Average::mean($scores);
    }
}
