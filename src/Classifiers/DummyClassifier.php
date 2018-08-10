<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\PopularityContest;
use InvalidArgumentException;

/**
 * Dummy Classifier
 *
 * A classifier that uses a user-defined Guessing Strategyto make predictions.
 * Dummy Classifier is useful to provide a sanity check and to compare
 * performance with an actual classifier.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DummyClassifier implements Estimator, Persistable
{
    /**
     * The guessing strategy that the dummy employs.
     *
     * @var \Rubix\ML\Other\Strategies\Categorical
     */
    protected $strategy;

    /**
     * @param  \Rubix\ML\Other\Strategies\Categorical  $strategy
     * @return void
     */
    public function __construct(Categorical $strategy = null)
    {
        if (is_null($strategy)) {
            $strategy = new PopularityContest();
        }

        $this->strategy = $strategy;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
    }

    /**
     * Fit the training set to the given guessing strategy.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $this->strategy->fit($dataset->labels());
    }

    /**
     * Make a prediction of a given sample dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = $this->strategy->guess();
        }

        return $predictions;
    }
}
