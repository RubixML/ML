<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function strlen;
use function strpos;
use function substr;
use function get_class;
use function hash_hmac;
use function hash_equals;
use function json_encode;
use function json_decode;
use function serialize;
use function unserialize;
use function gzencode;
use function gzdecode;
use function array_pad;
use function explode;

/**
 * RBX
 *
 * Rubix Object File format (RBX) is a format designed to securely and reliably store and share serialized PHP objects.
 * Based on PHP's native serialization format, RBX adds additional layers of compression, tamper protection, and class
 * compatibility detection all in one robust format.
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
     * The current version of the file format.
     *
     * @var int
     */
    protected const VERSION = 1;

    /**
     * The hashing function used to generate HMACs.
     *
     * @var string
     */
    protected const HASHING_FUNCTION = 'sha256';

    /**
     * The end of line character.
     *
     * @var string
     */
    protected const EOL = "\n";

    /**
     * The secret key used to sign and verify HMACs.
     *
     * @var string
     */
    protected $password;

    /**
     * The compression level between 0 and 9, 0 meaning no compression.
     *
     * @var int
     */
    protected $compressionLevel;

    /**
     * The base native serializer.
     *
     * @var \Rubix\ML\Persisters\Serializers\Native
     */
    protected $base;

    /**
     * @param string $password
     * @param int $compressionLevel
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(string $password = '', int $compressionLevel = 9)
    {
        if ($compressionLevel < 0 or $compressionLevel > 9) {
            throw new InvalidArgumentException('Compression level must'
                . " be between 0 and 9, $compressionLevel given.");
        }

        $this->password = $password;
        $this->compressionLevel = $compressionLevel;
        $this->base = new Native();
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

        $payload = gzencode($encoding, $this->compressionLevel);

        if ($payload === false) {
            throw new RuntimeException('Error compressing the data.');
        }

        $hash = hash_hmac(self::HASHING_FUNCTION, $payload, $this->password);

        $header = json_encode([
            'version' => self::VERSION,
            'class' => [
                'name' => get_class($persistable),
                'revision' => $persistable->revision(),
            ],
            'data' => [
                'hmac' => [
                    'type' => self::HASHING_FUNCTION,
                    'token' => $hash,
                ],
                'length' => strlen($payload),
            ],
        ]);

        if ($header === false) {
            throw new RuntimeException('Error encoding header.');
        }

        $hash = hash_hmac(self::HASHING_FUNCTION, $header, $this->password);

        $data = self::IDENTIFIER_STRING;
        $data .= self::HASHING_FUNCTION . ':' . $hash . self::EOL;
        $data .= $header . self::EOL;
        $data .= $payload;

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
            throw new RuntimeException('Unrecognized format.');
        }

        $data = substr($encoding, strlen(self::IDENTIFIER_STRING));

        [$hmac, $header, $payload] = array_pad(explode(self::EOL, $data, 3), 3, null);

        if (!$hmac or !$header or !$payload) {
            throw new RuntimeException('Invalid format.');
        }

        [$type, $token] = array_pad(explode(':', $hmac, 2), 2, null);

        $hash = hash_hmac($type, $header, $this->password);

        if (!hash_equals($hash, $token)) {
            throw new RuntimeException('Header failed verification.');
        }

        $header = json_decode($header);

        switch ($header->data->hmac->type) {
            case 'sha256':
                $hash = hash_hmac('sha256', $payload, $this->password);

                break;

            default:
                throw new RuntimeException('Invalid HMAC hash type.');
        }

        if (!hash_equals($hash, $header->data->hmac->token)) {
            throw new RuntimeException('Data failed verification.');
        }

        if (strlen($payload) !== $header->data->length) {
            throw new RuntimeException('Data is corrupted.');
        }

        $payload = gzdecode($payload);

        if ($payload === false) {
            throw new RuntimeException('Error decompressing the data.');
        }

        $persistable = $this->base->unserialize(new Encoding($payload));

        if (get_class($persistable) !== $header->class->name) {
            throw new RuntimeException('Class name mismatch.');
        }

        if ($persistable->revision() !== $header->class->revision) {
            throw new RuntimeException('Class revision number mismatch.');
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
        return "RBX (compression_level: {$this->compressionLevel})";
    }
}
