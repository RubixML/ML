<?php

namespace Rubix\ML\Traits;

use ReflectionClass;
use ReflectionNamedType;
use ReflectionProperty;
use SplObjectStorage;

use Throwable;

use function is_object;
use function is_array;
use function array_key_exists;
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

        $frames = [[$this, $this->persistableProperties($this), 0]];

        $seen[$this] = true;

        $tokens = [];

        while ($frames) {
            [$node, $properties, $index] = array_pop($frames);

            if ($index === count($properties)) {
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
                $frames[] = [$descend, $this->persistableProperties($descend), 0];

                $seen[$descend] = true;
            }
        }

        sort($tokens);

        return hash('crc32b', implode($tokens));
    }

    /**
     * Return the set of properties of the node that are included when the object is
     * serialized. Transient properties that are excluded from the serialized state are
     * omitted, so that the revision hash reflects only the persisted definition.
     *
     * @internal
     *
     * @param object $node
     * @return list<ReflectionProperty>
     */
    private function persistableProperties(object $node) : array
    {
        $reflector = new ReflectionClass($node);

        $properties = $reflector->getProperties();

        if (!$reflector->hasMethod('__serialize')) {
            return $properties;
        }

        try {
            $persisted = $reflector->getMethod('__serialize')->invoke($node);
        } catch (Throwable $error) {
            return $properties;
        }

        if (!is_array($persisted)) {
            return $properties;
        }

        $persistable = [];

        foreach ($properties as $property) {
            if (array_key_exists($property->getName(), $persisted)) {
                $persistable[] = $property;
            }
        }

        return $persistable;
    }
}
