<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Other\Helpers\JSON;
use Rubix\ML\Exceptions\ClassRevisionMismatch;
use Rubix\ML\Exceptions\RuntimeException;

use function strlen;
use function strpos;
use function substr;
use function hash;
use function get_class;
use function array_pad;
use function explode;

use const Rubix\ML\VERSION as LIBRARY_VERSION;

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
    protected const VERSION = 1;

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
     * The base serializer.
     *
     * @var \Rubix\ML\Persisters\Serializers\Gzip
     */
    protected $base;

    public function __construct()
    {
        $this->base = new Gzip(9, new Native());
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
            'library' => [
                'version' => LIBRARY_VERSION,
            ],
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

        $data = self::IDENTIFIER_STRING;
        $data .= self::VERSION . self::EOL;
        $data .= $checksum . self::EOL;
        $data .= $header . self::EOL;
        $data .= $encoding;

        return new Encoding($data);
    }

    /**
     * Unserialize a persistable object and return it.
     *
     * @internal
     *
     * @param \Rubix\ML\Encoding $encoding
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function unserialize(Encoding $encoding) : Persistable
    {
        if (strpos($encoding, self::IDENTIFIER_STRING) !== 0) {
            throw new RuntimeException('Unrecognized message format.');
        }

        $data = substr($encoding, strlen(self::IDENTIFIER_STRING));

        [$version, $checksum, $header, $payload] = array_pad(explode(self::EOL, $data, 4), 4, null);

        if (!$version or !$checksum or !$header or !$payload) {
            throw new RuntimeException('Invalid message format.');
        }

        [$type, $hash] = array_pad(explode(':', $checksum, 2), 2, null);

        if ($hash !== hash($type, $header)) {
            throw new RuntimeException('Header checksum verification failed.');
        }

        $header = JSON::decode($header);

        if (strlen($payload) !== $header['data']['length']) {
            throw new RuntimeException('Data is corrupted.');
        }

        $hash = hash($header['data']['checksum']['type'], $payload);

        if ($header['data']['checksum']['hash'] !== $hash) {
            throw new RuntimeException('Data checksum verification failed.');
        }

        $persistable = $this->base->unserialize(new Encoding($payload));

        if (get_class($persistable) !== $header['class']['name']) {
            throw new RuntimeException('Class name mismatch.');
        }

        if ($persistable->revision() !== $header['class']['revision']) {
            throw new ClassRevisionMismatch($header['library']['version']);
        }

        return $persistable;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'RBX';
    }
}
