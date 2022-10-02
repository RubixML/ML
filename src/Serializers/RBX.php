<?php

namespace Rubix\ML\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\ClassRevisionMismatch;
use Rubix\ML\Exceptions\RuntimeException;

use function strlen;
use function strpos;
use function substr;
use function hash;
use function get_class;
use function explode;

/**
 * RBX
 *
 * Rubix Object File format (RBX) is a format designed to reliably store and share serialized PHP objects. Based on PHP's native
 * serialization format, RBX adds additional layers of compression, data integrity checks, and class compatibility detection all
 * in one robust format.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RBX implements Serializer
{
    /**
     * The identifier or "magic number" of the format.
     *
     * @var string
     */
    protected const IDENTIFIER_STRING = "\241RBX\r\n\032\n";

    /**
     * The current version of the format.
     *
     * @var int
     */
    protected const VERSION = 2;

    /**
     * The hashing function used to generate checksums.
     *
     * @var string
     */
    protected const CHECKSUM_HASH_TYPE = 'crc32b';

    /**
     * The end of line character.
     *
     * @var string
     */
    protected const EOL = "\n";

    /**
     * The base Gzip Native serializer.
     *
     * @var \Rubix\ML\Serializers\GzipNative
     */
    protected \Rubix\ML\Serializers\GzipNative $base;

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
     * @param \Rubix\ML\Persistable $persistable
     * @return \Rubix\ML\Encoding
     */
    public function serialize(Persistable $persistable) : Encoding
    {
        $encoding = $this->base->serialize($persistable);

        $hash = hash(self::CHECKSUM_HASH_TYPE, $encoding);

        $header = JSON::encode([
            'class' => [
                'name' => get_class($persistable),
                'revision' => $persistable->revision(),
            ],
            'data' => [
                'checksum' => [
                    'type' => self::CHECKSUM_HASH_TYPE,
                    'hash' => $hash,
                ],
                'length' => $encoding->bytes(),
            ],
        ]);

        $hash = hash(self::CHECKSUM_HASH_TYPE, $header);

        $checksum = self::CHECKSUM_HASH_TYPE . ':' . $hash;

        $id = self::IDENTIFIER_STRING . self::VERSION;

        $data = $id . self::EOL;
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
     * @param \Rubix\ML\Encoding $encoding
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function deserialize(Encoding $encoding) : Persistable
    {
        [$version, $header, $payload] = $this->unpackMessage($encoding);

        if (strlen($payload) !== $header['data']['length']) {
            throw new RuntimeException('Data is corrupted.');
        }

        $hash = hash($header['data']['checksum']['type'], $payload);

        if ($header['data']['checksum']['hash'] !== $hash) {
            throw new RuntimeException('Data checksum verification failed.');
        }

        $persistable = $this->base->deserialize(new Encoding($payload));

        if (get_class($persistable) !== $header['class']['name']) {
            throw new RuntimeException('Class name mismatch.');
        }

        if ($persistable->revision() !== $header['class']['revision']) {
            throw new ClassRevisionMismatch();
        }

        return $persistable;
    }

    /**
     * Unpack the message version, checksum, header, and payload.
     *
     * @param \Rubix\ML\Encoding $encoding
     * @return array<mixed>
     */
    protected function unpackMessage(Encoding $encoding) : array
    {
        if (strpos($encoding, self::IDENTIFIER_STRING) !== 0) {
            throw new RuntimeException('Unrecognized message identifier.');
        }

        $data = substr($encoding, strlen(self::IDENTIFIER_STRING));

        $sections = explode(self::EOL, $data, 4);

        if (count($sections) !== 4) {
            throw new RuntimeException('Invalid message format.');
        }

        [$version, $checksum, $header, $payload] = $sections;

        if (!is_numeric($version)) {
            throw new RuntimeException('Invalid message format.');
        }

        $version = (int) $version;

        $checksum = explode(':', $checksum, 2);

        if (count($checksum) !== 2) {
            throw new RuntimeException('Invalid message format.');
        }

        [$type, $hash] = $checksum;

        if ($hash !== hash($type, $header)) {
            throw new RuntimeException('Header checksum verification failed.');
        }

        $header = JSON::decode($header);

        return [$version, $header, $payload];
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
        return "RBX (level: {$this->base->level()})";
    }
}
