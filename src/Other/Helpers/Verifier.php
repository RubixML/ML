<?php

namespace Rubix\ML\Other\Helpers;

use Rubix\ML\Specifications\Specification;
use InvalidArgumentException;

class Verifier
{
    /**
     * Eagerly check a list of specifications.
     *
     * @param iterable<\Rubix\ML\Specifications\Specification> $specifications
     */
    public static function check(iterable $specifications) : void
    {
        foreach ($specifications as $specification) {
            if (!$specification instanceof Specification) {
                throw new InvalidArgumentException('Invalid specification.');
            }

            $specification->check();
        }
    }
}
