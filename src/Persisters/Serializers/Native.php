<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use __PHP_Incomplete_Class;
use RuntimeException;
use Stringable;

use function is_object;

/**
 * Native
 *
 * The native PHP plain text serialization format.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Native implements Serializer, Stringable
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
     * Unserialize a persistable object and return it.
     *
     * @internal
     *
     * @param \Rubix\ML\Encoding $encoding
     * @throws RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(Encoding $encoding) : Persistable
    {
        $persistable = unserialize((string) $encoding);

        if ($persistable === false) {
            throw new RuntimeException('Cannot read encoding, wrong'
                . ' format or corrupted data.');
        }

        if (!is_object($persistable)) {
            throw new RuntimeException('Unserialized encoding must'
                . ' be an object.');
        }

        if ($persistable instanceof __PHP_Incomplete_Class) {
            throw new RuntimeException('Missing class definition'
                . ' for unserialized object.');
        }

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Unserialized object must'
                . ' implement the Persistable interface.');
        }

        return $persistable;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Native';
    }
}
