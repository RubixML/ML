<?php

namespace Rubix\Engine\CrossValidation;

use Rubix\Engine\Estimator;
use MathPHP\Statistics\Average;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Regressors\Regressor;
use Rubix\Engine\Clusterers\Clusterer;
use Rubix\Engine\Classifiers\Classifier;
use Rubix\Engine\Metrics\Validation\Validation;
use Rubix\Engine\Metrics\Validation\Regression;
use Rubix\Engine\Metrics\Validation\Clustering;
use Rubix\Engine\Metrics\Validation\Classification;
use InvalidArgumentException;

class KFold implements Validator
{
    /**
     * The metric used to score the predictions.
     *
     * @var \Rubix\Engine\Metrics\Validation
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
     * @param  \Rubix\Engine\Metrics\Validation\Validation  $metric
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
     * @param  \Rubix\Engine\Estimator\Estimator  $estimator
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @return float
     */
    public function score(Estimator $estimator, Labeled $dataset) : float
    {
        if ($estimator instanceof Classifier) {
            if (!$this->metric instanceof Classification) {
                throw new InvalidArgumentException('Classification metric only'
                    . ' works on Classifiers, ' . get_class($estimator)
                    . ' found.');
            }
        }

        if ($estimator instanceof Regressor) {
            if (!$this->metric instanceof Regression) {
                throw new InvalidArgumentException('Regression metric only'
                    . ' works on Regressors, ' . get_class($estimator)
                    . ' found.');
            }
        }

        if ($estimator instanceof Clusterer) {
            if (!$this->metric instanceof Clustering) {
                throw new InvalidArgumentException('Clustering metric only'
                    . ' works on Clusterers, ' . get_class($estimator)
                    . ' found.');
            }
        }

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

            $predictions = $estimator->predict($testing);

            $scores[] = $this->metric->score($predictions, $testing->labels());
        }

        return Average::mean($scores);
    }
}
