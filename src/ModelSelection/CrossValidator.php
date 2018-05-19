<?php

namespace Rubix\Engine\ModelSelection;

use MathPHP\Statistics\Average;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Estimator;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Regression;
use Rubix\Engine\Metrics\Validation\Classification;

class CrossValidator
{
    /**
     * The metric used to score the predictions.
     *
     * @var \Rubix\Engine\Metrics\Validation
     */
    protected $metric;

    /**
     * @param  \Rubix\Engine\Metrics\Validation\Validation  $metric
     * @return void
     */
    public function __construct(Validation $metric)
    {
        $this->metric = $metric;
    }

    /**
     * Run k training rounds where k is the number of folds. For each round use
     * one fold for testing and the rest to train the model. Return the average
     * score for each training round.
     *
     * @param  \Rubix\Engine\Estimator\Estimator  $estimator
     * @param  array  $folds
     * @return float
     */
    public function validate(Estimator $estimator, array $folds) : float
    {
        if ($estimator instanceof Classifier
            && !$this->metric instanceof Classfication) {
            throw new InvalidArgumentException('Validation metric only works on'
                . ' Classifiers, ' . get_class($metric) . ' found.');
        }

        if ($estimator instanceof Regressor
            && !$this->metric instanceof Regression) {
            throw new InvalidArgumentException('Validation metric only works on'
                . ' Regressors, ' . get_class($metric) . ' found.');
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

            $estimator->train(Supervised::combine($training));

            $predictions = $estimator->predict($testing);

            $scores[] = $this->metric->score($predictions, $testing->labels());
        }

        return Average::mean($scores);
    }
}
