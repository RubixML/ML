<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Persistable;
use RuntimeException;

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
class Igbinary implements Serializer
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
     * @return string
     */
    public function serialize(Persistable $persistable) : string
    {
        return igbinary_serialize($persistable) ?: '';
    }

    /**
     * Unserialize a persistable object and return it.
     *
     * @param string $data
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(string $data) : Persistable
    {
        return igbinary_unserialize($data);
    }
}
