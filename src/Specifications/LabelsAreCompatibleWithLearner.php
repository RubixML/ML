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
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected \Rubix\ML\Datasets\Labeled $dataset;

    /**
     * The learner instance.
     *
     * @var \Rubix\ML\Learner
     */
    protected \Rubix\ML\Learner $estimator;

    /**
     * Build a specification object with the given arguments.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Learner $estimator
     * @return self
     */
    public static function with(Labeled $dataset, Learner $estimator) : self
    {
        return new self($dataset, $estimator);
    }

    /**
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Learner $estimator
     */
    public function __construct(Labeled $dataset, Learner $estimator)
    {
        $this->dataset = $dataset;
        $this->estimator = $estimator;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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
