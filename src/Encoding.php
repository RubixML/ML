<?php

namespace Rubix\ML;

use Rubix\ML\Persisters\Persister;
use Rubix\ML\Serializers\Serializer;
use Stringable;

use function strlen;

class Encoding implements Stringable
{
    /**
     * The encoded data.
     *
     * @var string
     */
    protected string $data;

    /**
     * @param string $data
     */
    public function __construct(string $data)
    {
        $this->data = $data;
    }

    /**
     * Return the encoded data.
     *
     * @return string
     */
    public function data() : string
    {
        return $this->data;
    }

    /**
     * Deserialize the encoding with a given serializer and return a persistable object.
     *
     * @param \Rubix\ML\Serializers\Serializer $serializer
     * @return \Rubix\ML\Persistable
     */
    public function deserializeWith(Serializer $serializer) : Persistable
    {
        return $serializer->deserialize($this);
    }

    /**
     * Save the encoding with a given persister.
     *
     * @param \Rubix\ML\Persisters\Persister $persister
     */
    public function saveTo(Persister $persister) : void
    {
        $persister->save($this);
    }

    /**
     * Return the size of the encoding in bytes.
     *
     * @return int
     */
    public function bytes() : int
    {
        return strlen($this->data);
    }

    /**
     * Return the object as a string.
     *
     * @return string
     */
    public function __toString() : string
    {
        return $this->data;
    }
}
