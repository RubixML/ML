<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Strategies\Continuous;
use InvalidArgumentException;
use RuntimeException;

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
     * @return int[]
     */
    public function compatibility() : array
    {
        return DataType::ALL;
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
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        $this->strategy->fit($dataset->labels());

        $this->trained = true;
    }

    /**
     * Make a prediction of a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained) {
            throw new RuntimeException('The learner has not'
                . ' been trained.');
        }

        $n = $dataset->numRows();

        $predictions = [];

        for ($i = 0; $i < $n; $i++) {
            $predictions[] = $this->strategy->guess();
        }

        return $predictions;
    }
}
