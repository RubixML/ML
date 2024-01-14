<?php

namespace Rubix\ML\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Stringable;

/**
 * Serializer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Serializer extends Stringable
{
    /**
     * Serialize a persistable object and return the data.
     *
     * @internal
     *
     * @param Persistable $persistable
     * @return Encoding
     */
    public function serialize(Persistable $persistable) : Encoding;

    /**
     * Deserialize a persistable object and return it.
     *
     * @internal
     *
     * @param Encoding $encoding
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return Persistable
     */
    public function deserialize(Encoding $encoding) : Persistable;
}
