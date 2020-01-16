<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use InvalidArgumentException;

use function count;

class SamplesAreCompatibleWithEstimator
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Estimator $estimator
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Estimator $estimator) : void
    {
        $compatibility = $estimator->compatibility();

        $types = $dataset->uniqueTypes();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $diff = array_diff($types, $compatibility);

            $diffString = implode(', ', array_map([DataType::class, 'asString'], $diff));

            throw new InvalidArgumentException(Params::shortName($estimator)
                . " is not compatible with $diffString data types.");
        }
    }
}
