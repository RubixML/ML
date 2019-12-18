<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

class LabelsAreCompatibleWithLearner
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Datasets\Labeled<array> $dataset
     * @param \Rubix\ML\Learner $estimator
     * @throws \InvalidArgumentException
     */
    public static function check(Labeled $dataset, Learner $estimator) : void
    {
        $labelType = $dataset->labelType();

        switch ($estimator->type()) {
            case Estimator::CLASSIFIER:
                if ($labelType !== DataType::CATEGORICAL) {
                    throw new InvalidArgumentException('Classifiers require'
                        . ' categorical labels, ' . DataType::TYPES[$labelType]
                        . ' given.');
                }

                break 1;

            case Estimator::REGRESSOR:
                if ($labelType !== DataType::CONTINUOUS) {
                    throw new InvalidArgumentException('Regressors require'
                        . ' continuous labels, ' . DataType::TYPES[$labelType]
                        . ' given.');
                }

                break 1;
        }
    }
}
