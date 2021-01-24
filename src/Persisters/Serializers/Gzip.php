<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Gzip
 *
 * A compression format based on the DEFLATE algorithm with a header and CRC32 checksum.
 *
 * References:
 * [1] P. Deutsch. (1996). RFC 1951 - DEFLATE Compressed Data Format Specification version.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Gzip implements Serializer
{
    /**
     * The compression level between 0 and 9, 0 meaning no compression.
     *
     * @var int
     */
    protected $level;

    /**
     * The base serializer.
     *
     * @var \Rubix\ML\Persisters\Serializers\Serializer
     */
    protected $base;

    /**
     * @param int $level
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $base
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $level = 1, ?Serializer $base = null)
    {
        if ($level < 0 or $level > 9) {
            throw new InvalidArgumentException('Level must be'
                . " between 0 and 9, $level given.");
        }

        if ($base instanceof self) {
            throw new InvalidArgumentException('Base serializer'
                . ' must not be an instance of Gzip.');
        }

        $this->level = $level;
        $this->base = $base ?? new Native();
    }

    /**
     * Serialize a persistable object and return the data.
     *
     * @param \Rubix\ML\Persistable $persistable
     * @return \Rubix\ML\Encoding
     */
    public function serialize(Persistable $persistable) : Encoding
    {
        $encoding = $this->base->serialize($persistable);

        $data = gzencode($encoding, $this->level);

        if ($data === false) {
            throw new RuntimeException('Failed to compress data.');
        }

        return new Encoding($data);
    }

    /**
     * Unserialize a persistable object and return it.
     *
     * @param \Rubix\ML\Encoding $encoding
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(Encoding $encoding) : Persistable
    {
        $data = gzdecode($encoding);

        if ($data === false) {
            throw new RuntimeException('Failed to decompress data.');
        }

        return $this->base->unserialize(new Encoding($data));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Gzip (level: {$this->level}, base: {$this->base})";
    }
}
