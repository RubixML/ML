<?php

namespace Rubix\ML\Other\Traits;

use ReflectionClass;

use function is_object;
use function hash;
use function implode;
use function sort;

/**
 * Tracks Revisions
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait TracksRevisions
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

            foreach ($reflector->getProperties() as $property) {
                $tokens[] = $property->getName();
    
                $property->setAccessible(true);
    
                $value = $property->getValue($current);
    
                if (is_object($value)) {
                    $stack[] = $value;
                }
            }
        }

        sort($tokens);

        return hash('crc32b', implode($tokens));
    }
}
