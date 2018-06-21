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
     * The number of times to fold the dataset and therefore the number of
     * unique testing sets that will be generated.
     *
     * @var int
     */
    protected $folds;

    /**
     * @param  int  $folds
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $folds = 10)
    {
        if ($folds < 2) {
            throw new InvalidArgumentException('The number of folds cannot be'
                . ' less than two.');
        }

        $this->folds = $folds;
    }

    /**
     * Run k training rounds where k is the number of folds. For each round use
     * one fold for testing and the rest to train the model. Return the average
     * validation score for each training round.
     *
     * @param  \Rubix\ML\Estimator\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  \Rubix\ML\Metrics\Validation\Validation  $metric
     * @return float
     */
    public function test(Estimator $estimator, Labeled $dataset, Validation $metric) : float
    {
        if ($estimator instanceof Classifier or $estimator instanceof Clusterer) {
            $folds = $dataset->stratifiedFold($this->folds);
        } else {
            $folds = $dataset->fold($this->folds);
        }

        $scores = [];

        for ($i = 0; $i < $this->folds; $i++) {
            $training = [];

            for ($j = 0; $j < $this->folds; $j++) {
                if ($i === $j) {
                    $testing = clone $folds[$j];
                } else {
                    $training[] = clone $folds[$j];
                }
            }

            $estimator->train(Labeled::combine($training));

            $scores[] = $metric->score($estimator, $testing);
        }

        return Average::mean($scores);
    }
}
