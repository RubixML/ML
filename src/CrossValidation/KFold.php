<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Parallel;
use Rubix\ML\Estimator;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Other\Specifications\EstimatorIsCompatibleWithMetric;
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
class KFold implements Validator, Parallel
{
    use Multiprocessing;

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
     * @param int $k
     * @param bool $stratify
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 10, bool $stratify = false)
    {
        if ($k < 2) {
            throw new InvalidArgumentException('K cannot be less than 2'
                . ", $k given.");
        }

        $this->k = $k;
        $this->stratify = $stratify;
        $this->backend = new Serial();
    }

    /**
     * Test the estimator with the supplied dataset and return a score.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @throws \InvalidArgumentException
     * @return float
     */
    public function test(Learner $estimator, Labeled $dataset, Metric $metric) : float
    {
        EstimatorIsCompatibleWithMetric::check($estimator, $metric);

        $folds = $this->stratify
            ? $dataset->stratifiedFold($this->k)
            : $dataset->fold($this->k);

        $scores = [];

        for ($i = 0; $i < $this->k; $i++) {
            $training = Labeled::quick();
            $testing = Labeled::quick();
    
            foreach ($folds as $j => $fold) {
                if ($i === $j) {
                    $testing = $testing->append($fold);
                } else {
                    $training = $training->append($fold);
                }
            }
            
            $this->backend->enqueue(
                [self::class, 'score'],
                [$estimator, $training, $testing, $metric]
            );
        }

        $scores = $this->backend->process();

        return Stats::mean($scores);
    }

    /**
     * Score an estimator on one of k folds of the dataset.
     *
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Datasets\Dataset $training
     * @param \Rubix\ML\Datasets\Labeled $testing
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @return float
     */
    public static function score(Learner $estimator, Dataset $training, Labeled $testing, Metric $metric) : float
    {
        $estimator->train($training);

        $predictions = $estimator->predict($testing);

        return $metric->score($predictions, $testing->labels());
    }
}
