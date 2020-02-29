<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Embedders\Embedder;
use Rubix\ML\Other\Helpers\Params;
use InvalidArgumentException;

use function count;
use function get_class;

class SamplesAreCompatibleWithEmbedder
{
    /**
     * Perform a check of the specification.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Embedders\Embedder $embedder
     * @throws \InvalidArgumentException
     */
    public static function check(Dataset $dataset, Embedder $embedder) : void
    {
        $compatibility = $embedder->compatibility();

        $types = $dataset->uniqueTypes();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $diff = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                Params::shortName(get_class($embedder))
                . ' is only compatible with '
                . implode(', ', $compatibility) . ' data types, '
                . implode(', ', $diff) . ' given.'
            );
        }
    }
}
