<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Persistable;

/**
 * Native
 *
 * The native PHP plain text serialization format.
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
     * @param \Rubix\ML\Persistable $persistable
     * @return string
     */
    public function serialize(Persistable $persistable) : string
    {
        return serialize($persistable);
    }

    /**
     * Unserialize a persistable object and return it.
     *
     * @param string $data
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(string $data) : Persistable
    {
        return unserialize($data);
    }
}
