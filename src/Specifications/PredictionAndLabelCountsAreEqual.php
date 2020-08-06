<?php

namespace Rubix\ML\Specifications;

use InvalidArgumentException;

class PredictionAndLabelCountsAreEqual
{
    /**
     * Perform a check of the specification.
     *
     * @param (string|int|float)[] $predictions
     * @param (string|int|float)[] $labels
     * @throws \InvalidArgumentException
     */
    public static function check(array $predictions, array $labels) : void
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException(
                'Number of predictions'
                . ' and labels must be equal, ' . count($predictions)
                . ' predictions and ' . count($labels) . ' labels given.'
            );
        }
    }
}
