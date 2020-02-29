<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;

use function count;
use function get_class;

class SamplesAreCompatibleWithDistance
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Distance $kernel) : void
    {
        $compatibility = $kernel->compatibility();

        $types = $dataset->uniqueTypes();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $diff = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                Params::shortName(get_class($kernel))
                . ' is only compatible with '
                . implode(', ', $compatibility) . ' data types, '
                . implode(', ', $diff) . ' given.'
            );
        }
    }
}
