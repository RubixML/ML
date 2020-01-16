<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Learner;
use Rubix\ML\Deferred;
use Rubix\ML\Parallel;
use Rubix\ML\DataType;
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
 * K Fold is a cross validation technique that splits the training set into *k* individual
 * folds and for each training round uses 1 of the folds to test the model and the rest as
 * training data. The final score is the average validation score over all of the *k*
 * rounds. K Fold has the advantage of both training and testing on each sample in the
 * dataset at least once.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KFold implements Validator, Parallel
{
    use Multiprocessing;

    /**
     * The number of folds to split the dataset into.
     *
     * @var int
     */
    protected $k;

    /**
     * @param int $k
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 5)
    {
        if ($k < 2) {
            throw new InvalidArgumentException("K cannot be less than 2, $k given.");
        }

        $this->k = $k;
        $this->backend = new Serial();
    }

    /**
     * Test the estimator with the supplied dataset and return a validation score.
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

        $dataset->randomize();

        $folds = $dataset->labelType() === DataType::CATEGORICAL
            ? $dataset->stratifiedFold($this->k)
            : $dataset->fold($this->k);

        $this->backend->flush();

        for ($i = 0; $i < $this->k; ++$i) {
            $training = Labeled::quick();
            $testing = Labeled::quick();
    
            foreach ($folds as $j => $fold) {
                if ($i === $j) {
                    $testing = $testing->append($fold);
                } else {
                    $training = $training->append($fold);
                }
            }
            
            $this->backend->enqueue(new Deferred(
                [self::class, 'score'],
                [$estimator, $training, $testing, $metric]
            ));
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
