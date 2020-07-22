<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;

interface Serializer
{
    /**
     * Serialize a persistable object and return the data.
     *
     * @param \Rubix\ML\Persistable $persistable
     * @return \Rubix\ML\Encoding
     */
    public function serialize(Persistable $persistable) : Encoding;

    /**
     * Unserialize a persistable object and return it.
     *
     * @param \Rubix\ML\Encoding $data
     * @throws \RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(Encoding $data) : Persistable;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
