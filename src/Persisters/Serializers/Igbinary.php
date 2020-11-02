<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use __PHP_Incomplete_Class;
use RuntimeException;
use Stringable;

use function is_null;
use function is_object;

/**
 * Igbinary
 *
 * Igbinary is a compact binary format that serves as a drop-in replacement for the native PHP
 * serializer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Igbinary implements Serializer, Stringable
{
    /**
     * @throws \RuntimeException
     */
    public function __construct()
    {
        if (!extension_loaded('igbinary')) {
            throw new RuntimeException('Igbinary extension is not loaded,'
                . ' check PHP configuration.');
        }
    }

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
        return new Encoding(igbinary_serialize($persistable) ?: '');
    }

    /**
     * Unserialize a persistable object and return it.
     *
     * @internal
     *
     * @param \Rubix\ML\Encoding $encoding
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(Encoding $encoding) : Persistable
    {
        $persistable = igbinary_unserialize((string) $encoding);

        if (is_null($persistable)) {
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
        return 'Igbinary';
    }
}
