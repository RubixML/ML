<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Strategies\Continuous;
use Rubix\ML\Other\Specifications\LabelsAreCompatibleWithLearner;
use InvalidArgumentException;
use RuntimeException;

use function count;

/**
 * Dummy Regressor
 *
 * Regressor that guesses the output values based on a Guessing Strategy. Dummy
 * Regressor is useful to provide a sanity check and to compare performance
 * against actual Regressors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DummyRegressor implements Estimator, Learner, Persistable
{
    use PredictsSingle;
    
    /**
     * The guessing strategy that the dummy employs.
     *
     * @var \Rubix\ML\Other\Strategies\Continuous
     */
    protected $strategy;

    /**
     * Has the learner been trained?
     *
     * @var bool
     */
    protected $trained;

    /**
     * @param \Rubix\ML\Other\Strategies\Continuous|null $strategy
     */
    public function __construct(?Continuous $strategy = null)
    {
        $this->strategy = $strategy ?? new Mean();
        $this->trained = false;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return DataType::all();
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

        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $this->strategy->fit($dataset->labels());

        $this->trained = true;
    }

    /**
     * Make a prediction of a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return (int|float)[]
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
