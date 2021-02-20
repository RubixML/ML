<?php

namespace Rubix\ML\Other\Traits;

use ReflectionClass;

use function is_object;
use function array_pop;
use function hash;
use function implode;
use function sort;

/**
 * Autotrack Revisions
 *
 * Automatically update class revision numbers by tracking changes in the object-property definition tree stemming from this instance.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait AutotrackRevisions
{
    /**
     * Return the revision number of the class.
     *
     * @return string
     */
    public function revision() : string
    {
        $stack = [$this];

        $tokens = [];

        while ($current = array_pop($stack)) {
            $reflector = new ReflectionClass($current);

            $properties = $reflector->getProperties();

            foreach ($properties as $property) {
                $property->setAccessible(true);

                $value = $property->getValue($current);

                if (is_object($value)) {
                    $stack[] = $value;
                }

                $tokens[] = $property->getName();
            }
        }

        sort($tokens);

        return hash('crc32b', implode($tokens));
    }
}
