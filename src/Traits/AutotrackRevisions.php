<?php

namespace Rubix\ML\Traits;

use ReflectionClass;
use ReflectionNamedType;

use function is_object;
use function array_pop;
use function hash;
use function implode;
use function sort;

/**
 * Autotrack Revisions
 *
 * Automatically update class revision hashes by tracking changes to the object-property definition
 * tree stemming from this instance.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait AutotrackRevisions
{
    /**
     * Return the class revision hash by traversing the object-property definition tree in depth-first
     * order.
     *
     * @return string
     */
    public function revision() : string
    {
        $stack = [$this];

        $tokens = [];

        while ($stack) {
            $current = array_pop($stack);

            $reflector = new ReflectionClass($current);

            $properties = $reflector->getProperties();

            foreach ($properties as $property) {
                $property->setAccessible(true);

                if ($property->isInitialized($current)) {
                    $value = $property->getValue($current);

                    if (is_object($value)) {
                        $stack[] = $value;
                    }

                    $type = $property->getType();

                    if ($type instanceof ReflectionNamedType) {
                        $type = $type->getName();
                    } else {
                        $type = 'mixed';
                    }

                    $name = $property->getName();

                    $tokens[] = "$type:$name";
                }
            }
        }

        sort($tokens);

        return hash('crc32b', implode($tokens));
    }
}
