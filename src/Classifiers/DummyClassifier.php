<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\Prior;
use InvalidArgumentException;
use RuntimeException;

use function count;

/**
 * Dummy Classifier
 *
 * A classifier that uses a user-defined Guessing Strategy to make predictions.
 * Dummy Classifier is useful to provide a sanity check and to compare
 * performance with an actual classifier.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DummyClassifier implements Estimator, Learner, Persistable
{
    use PredictsSingle;
    
    /**
     * The guessing strategy that the dummy employs.
     *
     * @var \Rubix\ML\Other\Strategies\Categorical
     */
    protected $strategy;

    /**
     * Has the learner been trained?
     *
     * @var bool
     */
    protected $trained;

    /**
     * @param \Rubix\ML\Other\Strategies\Categorical|null $strategy
     */
    public function __construct(?Categorical $strategy = null)
    {
        $this->strategy = $strategy ?? new Prior();
        $this->trained = false;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->trained;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return DataType::ALL;
    }

    /**
     * Fit the training set to the given guessing strategy.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' labeled training set.');
        }

        $this->strategy->fit($dataset->labels());

        $this->trained = true;
    }

    /**
     * Make a prediction of a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained) {
            throw new RuntimeException('Estimator has not been trained.');
        }
        
        $n = $dataset->numRows();

        $predictions = [];

        while (count($predictions) < $n) {
            $predictions[] = $this->strategy->guess();
        }

        return $predictions;
    }
}
