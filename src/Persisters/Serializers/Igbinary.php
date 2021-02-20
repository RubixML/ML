<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use __PHP_Incomplete_Class;

use function Rubix\ML\warn_deprecated;
use function is_object;

/**
 * Igbinary
 *
 * Igbinary is a compact binary format that serves as a drop-in replacement for the native PHP serializer.
 *
 * @deprecated
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Igbinary implements Serializer
{
    public function __construct()
    {
        warn_deprecated('Igbinary is deprecated, will move to Extras package in the next major release.');

        ExtensionIsLoaded::with('igbinary')->check();
    }

    /**
     * Serialize a persistable object and return the data.
     *
     * @internal
     *
     * @param \Rubix\ML\Persistable $persistable
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Encoding
     */
    public function serialize(Persistable $persistable) : Encoding
    {
        $data = igbinary_serialize($persistable);

        if (!$data) {
            throw new RuntimeException('Could not serialize data.');
        }

        return new Encoding($data);
    }

    /**
     * Unserialize a persistable object and return it.
     *
     * @internal
     *
     * @param \Rubix\ML\Encoding $encoding
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(Encoding $encoding) : Persistable
    {
        $persistable = igbinary_unserialize($encoding);

        if (!is_object($persistable)) {
            throw new RuntimeException('Unserialized data must be an object.');
        }

        if ($persistable instanceof __PHP_Incomplete_Class) {
            throw new RuntimeException('Missing class for object data.');
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
