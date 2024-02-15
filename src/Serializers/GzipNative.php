<?php

namespace Rubix\ML\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Gzip Native
 *
 * Gzip Native wraps the native PHP serialization format in an outer compression layer based on the
 * DEFLATE algorithm with a header and CRC32 checksum.
 *
 * References:
 * [1] P. Deutsch. (1996). RFC 1951 - DEFLATE Compressed Data Format Specification version.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GzipNative implements Serializer
{
    /**
     * The compression level between 0 and 9, 0 meaning no compression.
     *
     * @var int
     */
    protected int $level;

    /**
     * The base serializer.
     *
     * @var Native
     */
    protected Native $base;

    /**
     * @param int $level
     * @throws InvalidArgumentException
     */
    public function __construct(int $level = 6)
    {
        if ($level < 0 or $level > 9) {
            throw new InvalidArgumentException('Level must be'
                . " between 0 and 9, $level given.");
        }

        $this->level = $level;
        $this->base = new Native();
    }

    /**
     * Return the level of compression between 0 and 9.
     *
     * @internal
     *
     * @return int
     */
    public function level() : int
    {
        return $this->level;
    }

    /**
     * Serialize a persistable object and return the data.
     *
     * @param Persistable $persistable
     * @return Encoding
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
     * Deserialize a persistable object and return it.
     *
     * @param Encoding $encoding
     * @throws RuntimeException
     * @return Persistable
     */
    public function deserialize(Encoding $encoding) : Persistable
    {
        $data = gzdecode($encoding);

        if ($data === false) {
            throw new RuntimeException('Failed to decompress data.');
        }

        return $this->base->deserialize(new Encoding($data));
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Gzip (level: {$this->level})";
    }
}
