<?php

namespace Rubix\ML\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function Rubix\ML\warn;
use function strlen;
use function substr;
use function hash;
use function get_class;
use function array_pad;
use function explode;
use function gzencode;
use function gzdecode;

use const Rubix\ML\VERSION as LIBRARY_VERSION;

/**
 * RBX V1
 *
 * Rubix Object File format (RBXV1) is a format designed to reliably store and share serialized PHP objects. Based on PHP's native
 * serialization format, RBXV1 adds additional layers of compression, data integrity checks, and class compatibility detection all
 * in one robust format.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RBXV1 implements Serializer
{
    /**
     * The identifier or "magic number" of the format.
     *
     * @var string
     */
    protected const IDENTIFIER_STRING = "\241RBX\r\n\032\n";

    /**
     * The version of the format.
     *
     * @var int
     */
    protected const VERSION = 1;

    /**
     * The hashing function used to generate checksums.
     *
     * @var string
     */
    protected const CHECKSUM_TYPE = 'crc32b';

    /**
     * The end of line character.
     *
     * @var string
     */
    protected const EOL = "\n";

    /**
     * The level of gzip compression.
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
     * Serialize a persistable object and return the data.
     *
     * @internal
     *
     * @param Persistable $persistable
     * @return Encoding
     */
    public function serialize(Persistable $persistable) : Encoding
    {
        $className = get_class($persistable);

        $encoding = $this->base->serialize($persistable);

        $data = gzencode($encoding, $this->level);

        if ($data === false) {
            throw new RuntimeException('Failed to compress data.');
        }

        $encoding = new Encoding($data);

        $hash = hash(self::CHECKSUM_TYPE, $encoding);

        $header = JSON::encode([
            'library' => [
                'version' => LIBRARY_VERSION,
            ],
            'class' => [
                'name' => $className,
                'revision' => $persistable->revision(),
            ],
            'data' => [
                'checksum' => [
                    'type' => self::CHECKSUM_TYPE,
                    'hash' => $hash,
                ],
                'length' => $encoding->bytes(),
            ],
        ]);

        $hash = hash(self::CHECKSUM_TYPE, $header);

        $checksum = self::CHECKSUM_TYPE . ':' . $hash;

        $data = self::IDENTIFIER_STRING;
        $data .= self::VERSION . self::EOL;
        $data .= $checksum . self::EOL;
        $data .= $header . self::EOL;
        $data .= $encoding;

        return new Encoding($data);
    }

    /**
     * Deserialize a persistable object and return it.
     *
     * @internal
     *
     * @param Encoding $encoding
     * @throws RuntimeException
     * @return Persistable
     */
    public function deserialize(Encoding $encoding) : Persistable
    {
        if (!str_starts_with($encoding, self::IDENTIFIER_STRING)) {
            throw new RuntimeException('Unrecognized message format.');
        }

        $data = substr($encoding, strlen(self::IDENTIFIER_STRING));

        [$version, $checksum, $header, $payload] = array_pad(explode(self::EOL, $data, 4), 4, null);

        if (empty($version) or empty($checksum) or empty($header) or empty($payload)) {
            throw new RuntimeException('Invalid message format.');
        }

        if ($version != self::VERSION) {
            throw new RuntimeException('Incompatible version format, use the'
                . " RBX V{$version} serializer instead.");
        }

        [$type, $hash] = array_pad(explode(':', $checksum, 2), 2, null);

        if (empty($type) or empty($hash)) {
            throw new RuntimeException('Invalid header digest.');
        }

        if ($type != self::CHECKSUM_TYPE) {
            throw new RuntimeException('Invalid header checksum type.');
        }

        if (hash($type, $header) !== $hash) {
            throw new RuntimeException('Header checksum verification failed.');
        }

        $header = JSON::decode($header);

        $className = $header['class']['name'] ?? null;
        $revision = $header['class']['revision'] ?? null;
        $type = $header['data']['checksum']['type'] ?? null;
        $hash = $header['data']['checksum']['hash'] ?? null;
        $length = $header['data']['length'] ?? null;

        if (strlen($payload) !== $length) {
            throw new RuntimeException('Data length does not match header.');
        }

        if ($type != self::CHECKSUM_TYPE) {
            throw new RuntimeException('Invalid data checksum type.');
        }

        if (hash($type, $payload) !== $hash) {
            throw new RuntimeException('Data checksum verification failed.');
        }

        $data = gzdecode($payload);

        if ($data === false) {
            throw new RuntimeException('Failed to decompress data.');
        }

        $persistable = $this->base->deserialize(new Encoding($data));

        if ($persistable->revision() !== $revision) {
            warn("Class revision mismatch, expected $revision but"
                . " got {$persistable->revision()}. ");
        }

        if (get_class($persistable) !== $className) {
            throw new RuntimeException('Class name mismatch.');
        }

        return $persistable;
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
        return "RBX V1 (level: {$this->level})";
    }
}
