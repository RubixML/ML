<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use InvalidArgumentException;

use function count;
use function get_class;

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

            throw new InvalidArgumentException(
                Params::shortName(get_class($estimator))
                . ' is only compatible with '
                . implode(', ', $compatibility) . ' data types, '
                . implode(', ', $diff) . ' given.'
            );
        }
    }
}
