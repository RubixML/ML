<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Transformers\Transformer;
use InvalidArgumentException;

use function count;

class SamplesAreCompatibleWithTransformer
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Transformers\Transformer $transformer
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Transformer $transformer) : void
    {
        $compatibility = $transformer->compatibility();

        $types = $dataset->uniqueTypes();

        $compatible = array_intersect($types, $compatibility);

        if (count($compatible) < count($types)) {
            $incompatible = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                "$transformer is not compatible with " . implode(', ', $incompatible) . ' data types.'
            );
        }
    }
}
