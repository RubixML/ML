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
     * The validation score of each fold since the last test.
     * 
     * @var array|null
     */
    protected $scores;

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
     * Return the validation scores computed at last test time.
     * 
     * @return array|null
     */
    public function scores() : ?array
    {
        return $this->scores;
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

            foreach ($folds as $j => $fold) {
                if ($i === $j) {
                    $testing = clone $fold;
                } else {
                    $training = $training->merge($fold);
                }
            }

            $estimator->train($training);

            $scores[] = $metric->score($estimator, $testing);
        }

        $this->scores = $scores;

        return Stats::mean($scores);
    }
}
