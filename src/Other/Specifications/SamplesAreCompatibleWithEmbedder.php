<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Embedders\Embedder;
use Rubix\ML\Other\Helpers\Params;
use InvalidArgumentException;

use function count;

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
        $types = $dataset->uniqueTypes();

        $compatibility = $embedder->compatibility();

        $same = array_intersect($types, $compatibility);

        if (count($same) < count($types)) {
            $diff = array_diff($types, $compatibility);

            $diffString = implode(', ', $diff);

            throw new InvalidArgumentException(Params::shortName($embedder)
                . " is not compatible with $diffString data types.");
        }
    }
}
