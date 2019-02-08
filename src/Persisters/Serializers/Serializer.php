<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Persistable;

interface Serializer
{
    /**
     * Serialize a persistable object and return the data.
     *
     * @param \Rubix\ML\Persistable $persistable
     * @return string
     */
    public function serialize(Persistable $persistable) : string;

    /**
     * Unserialize a persistable object and return it.
     *
     * @param string $data
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(string $data) : Persistable;
}
