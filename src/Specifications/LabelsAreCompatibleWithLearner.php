<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * @internal
 */
class LabelsAreCompatibleWithLearner extends Specification
{
    /**
     * The dataset that contains the labels under validation.
     *
     * @var Labeled
     */
    protected Labeled $dataset;

    /**
     * The learner instance.
     *
     * @var Learner
     */
    protected Learner $estimator;

    /**
     * Build a specification object with the given arguments.
     *
     * @param Labeled $dataset
     * @param Learner $estimator
     * @return self
     */
    public static function with(Labeled $dataset, Learner $estimator) : self
    {
        return new self($dataset, $estimator);
    }

    /**
     * @param Labeled $dataset
     * @param Learner $estimator
     */
    public function __construct(Labeled $dataset, Learner $estimator)
    {
        $this->dataset = $dataset;
        $this->estimator = $estimator;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws InvalidArgumentException
     */
    public function check() : void
    {
        switch ($this->estimator->type()) {
            case EstimatorType::classifier():
                if ($this->dataset->labelType() != DataType::categorical()) {
                    throw new InvalidArgumentException(
                        'Classifiers require categorical labels,'
                        . " {$this->dataset->labelType()} given."
                    );
                }

                break;

            case EstimatorType::regressor():
                if ($this->dataset->labelType() != DataType::continuous()) {
                    throw new InvalidArgumentException(
                        'Regressors require continuous labels,'
                        . " {$this->dataset->labelType()} given."
                    );
                }

                break;
        }
    }
}
