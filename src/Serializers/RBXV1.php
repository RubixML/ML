<?php

namespace Rubix\ML\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\warn;
use function strlen;
use function substr;
use function hash;
use function get_class;
use function array_pad;
use function explode;

use const Rubix\ML\VERSION as LIBRARY_VERSION;

/**
 * RBXV1 V1
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
     * The base Gzip Native serializer.
     *
     * @var GzipNative
     */
    protected GzipNative $base;

    /**
     * @param int $level
     */
    public function __construct(int $level = 6)
    {
        $this->base = new GzipNative($level);
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
                . " RBXV1 V{$version} serializer instead.");
        }

        [$type, $hash] = array_pad(explode(':', $checksum, 2), 2, null);

        if (empty($type) or empty($hash)) {
            throw new RuntimeException('Invalid header digest.');
        }

        if ($type != self::CHECKSUM_TYPE) {
            throw new RuntimeException('Invalid header checksum type.');
        }

        if ($hash !== hash($type, $header)) {
            throw new RuntimeException('Header checksum verification failed.');
        }

        $header = JSON::decode($header);

        $length = $header['data']['length'] ?? null;

        if (strlen($payload) !== $length) {
            throw new RuntimeException('Data length does not match header.');
        }

        $dataChecksumType = $header['data']['checksum']['type'] ?? null;

        if ($dataChecksumType != self::CHECKSUM_TYPE) {
            throw new RuntimeException('Invalid data checksum type.');
        }

        $hash = hash($dataChecksumType, $payload);

        $expectedHash = $header['data']['checksum']['hash'] ?? null;

        if ($hash !== $expectedHash) {
            throw new RuntimeException('Data checksum verification failed.');
        }

        $persistable = $this->base->deserialize(new Encoding($payload));

        $expectedRevision = $header['class']['revision'] ?? null;

        if ($persistable->revision() !== $expectedRevision) {
            warn("Class revision mismatch, expected $expectedRevision but"
                . " got {$persistable->revision()}. ");
        }

        $className = $header['class']['name'] ?? null;

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
        return "RBXV1 V1 (level: {$this->base->level()})";
    }
}
