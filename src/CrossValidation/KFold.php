<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\CrossValidation\Metrics\Metric;
use InvalidArgumentException;

/**
 * K Fold
 *
 * In k-fold cross-validation, the dataset is partitioned into k equal sized
 * subsets. Of the k subsets, a single fold is retained as the validation set
 * for testing the model, and the remaining k − 1 subsets are used as training
 * data. The cross-validation process is then repeated k times, with each of the
 * k folds used exactly once as the validation data. The k results are then
 * averaged to produce a single validation score. The advantage of this method
 * over Hold Out is that all observations are used for both training and
 * validation, and each observation is used for validation exactly once.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KFold implements Validator
{
    /**
     * The number of times to fold the dataset and therefore the number of
     * unique testing sets that will be generated.
     *
     * @var int
     */
    protected $k;

    /**
     * Should we stratify the dataset before folding?
     *
     * @var bool
     */
    protected $stratify;

    /**
     * @param  int  $k
     * @param  bool  $stratify
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 10, bool $stratify = false)
    {
        if ($k < 2) {
            throw new InvalidArgumentException('The number of folds cannot be'
                . ' less than two.');
        }

        $this->k = $k;
        $this->stratify = $stratify;
    }

    /**
     * Test the estimator with the supplied dataset and return a score.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  \Rubix\ML\CrossValidation\Metrics\Metric  $metric
     * @return float
     */
    public function test(Estimator $estimator, Labeled $dataset, Metric $metric) : float
    {
        if ($this->stratify === true) {
            $folds = $dataset->stratifiedFold($this->k);
        } else {
            $folds = $dataset->randomize()->fold($this->k);
        }

        $scores = [];

        for ($i = 0; $i < $this->k; $i++) {
            $training = new Labeled();
            $testing = null;

            for ($j = 0; $j < $this->k; $j++) {
                if ($i === $j) {
                    $testing = clone $folds[$j];
                } else {
                    $training = $training->merge($folds[$j]);
                }
            }

            $estimator->train($training);

            $scores[] = $metric->score($estimator, $testing);
        }

        return Stats::mean($scores);
    }
}
