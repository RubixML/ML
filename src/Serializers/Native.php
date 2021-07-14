<?php

namespace Rubix\ML\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Exceptions\RuntimeException;
use __PHP_Incomplete_Class;

use function is_object;
use function serialize;
use function unserialize;

/**
 * Native
 *
 * The native bytecode format that comes bundled with PHP core.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Native implements Serializer
{
    /**
     * Serialize a persistable object and return the data.
     *
     * @internal
     *
     * @param \Rubix\ML\Persistable $persistable
     * @return \Rubix\ML\Encoding
     */
    public function serialize(Persistable $persistable) : Encoding
    {
        return new Encoding(serialize($persistable));
    }

    /**
     * Deserialize a persistable object and return it.
     *
     * @internal
     *
     * @param \Rubix\ML\Encoding $encoding
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function deserialize(Encoding $encoding) : Persistable
    {
        $persistable = unserialize($encoding);

        if (!is_object($persistable)) {
            throw new RuntimeException('Deserialized data must be an object.');
        }

        if ($persistable instanceof __PHP_Incomplete_Class) {
            throw new RuntimeException('Missing class for object data.');
        }

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Deserialized object must'
                . ' implement the Persistable interface.');
        }

        return $persistable;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Native';
    }
}
