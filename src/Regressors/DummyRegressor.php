<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

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
    use AutotrackRevisions;

    /**
     * The guessing strategy that the dummy employs.
     *
     * @var \Rubix\ML\Other\Strategies\Strategy
     */
    protected $strategy;

    /**
     * @param \Rubix\ML\Other\Strategies\Strategy|null $strategy
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(?Strategy $strategy = null)
    {
        if ($strategy and !$strategy->type()->isContinuous()) {
            throw new InvalidArgumentException('Continuous strategy must'
                . ' be compatible with continuous data types.');
        }

        $this->strategy = $strategy ?? new Mean();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'strategy' => $this->strategy,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->strategy->fitted();
    }

    /**
     * Fit the training set to the given guessing strategy.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        $this->strategy->fit($dataset->labels());
    }

    /**
     * Make a prediction of a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<int|float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->strategy->fitted()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return int|float
     */
    public function predictSample(array $sample)
    {
        /** @var int|float $prediction */
        $prediction = $this->strategy->guess();

        return $prediction;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Dummy Regressor (' . Params::stringify($this->params()) . ')';
    }
}
