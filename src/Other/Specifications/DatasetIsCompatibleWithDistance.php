<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Kernels\Distance\Distance;
use InvalidArgumentException;

class DatasetIsCompatibleWithDistance
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
        $types = $dataset->uniqueTypes();

        $compatibility = $kernel->compatibility();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $diff = array_diff($types, $compatibility);

            $diffString = implode(', ', array_map([DataType::class, 'asString'], $diff));

            throw new InvalidArgumentException(Params::shortName($kernel)
                . " is not compatible with $diffString data types.");
        }
    }
}
