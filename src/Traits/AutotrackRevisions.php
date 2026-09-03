<?php

namespace Rubix\ML\Traits;

use ReflectionClass;
use ReflectionNamedType;
use SplObjectStorage;

use function is_object;
use function array_pop;
use function count;
use function hash;
use function implode;
use function sort;

/**
 * Autotrack Revisions
 *
 * Automatically update class revision hashes by tracking changes to the object-property definition
 * tree stemming from this instance. Circular references are tolerated: a property whose value
 * points at an object already on the active traversal path is treated as a back-edge and
 * skipped, so the traversal always terminates.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait AutotrackRevisions
{
    /**
     * Return the class revision hash by traversing the object-property definition tree in
     * depth-first order.
     *
     * @return string
     */
    public function revision() : string
    {
        $seen = new SplObjectStorage();

        $reflector = new ReflectionClass($this);

        $frames = [[$this, $reflector->getProperties(), 0]];

        $seen[$this] = true;

        $tokens = [];

        while ($frames) {
            [$node, $properties, $index] = array_pop($frames);

            $total = count($properties);

            if ($index === $total) {
                unset($seen[$node]);

                continue;
            }

            $property = $properties[$index];

            $descend = null;

            if ($property->isInitialized($node)) {
                $value = $property->getValue($node);

                $type = $property->getType();

                if ($type instanceof ReflectionNamedType) {
                    $type = $type->getName();
                } else {
                    $type = 'mixed';
                }

                $name = $property->getName();

                $tokens[] = "{$type}:{$name}";

                if (is_object($value) and !isset($seen[$value])) {
                    $descend = $value;
                }
            }

            $frames[] = [$node, $properties, $index + 1];

            if ($descend) {
                $reflector = new ReflectionClass($descend);

                $frames[] = [$descend, $reflector->getProperties(), 0];

                $seen[$descend] = true;
            }
        }

        sort($tokens);

        return hash('crc32b', implode($tokens));
    }
}
