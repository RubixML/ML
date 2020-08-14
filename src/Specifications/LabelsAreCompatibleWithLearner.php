<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

class LabelsAreCompatibleWithLearner extends Specification
{
    /**
     * The dataset that contains the labels under validation.
     *
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected $dataset;

    /**
     * The learner.
     *
     * @var \Rubix\ML\Learner
     */
    protected $estimator;

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
     * @throws \InvalidArgumentException
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

                break 1;

            case EstimatorType::regressor():
                if ($this->dataset->labelType() != DataType::continuous()) {
                    throw new InvalidArgumentException(
                        'Regressors require continuous labels,'
                        . " {$this->dataset->labelType()} given."
                    );
                }

                break 1;
        }
    }
}
