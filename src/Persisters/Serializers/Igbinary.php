<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use RuntimeException;
use Stringable;

/**
 * Igbinary
 *
 * Igbinary is a compact binary format that serves as a drop-in replacement for the
 * native PHP serializer.
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
     * @param \Rubix\ML\Encoding $data
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(Encoding $data) : Persistable
    {
        return igbinary_unserialize($data);
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
