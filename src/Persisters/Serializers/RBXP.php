<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Other\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;

use function strlen;
use function strpos;
use function substr;
use function get_class;
use function hash_hmac;
use function hash_equals;
use function serialize;
use function unserialize;
use function array_pad;
use function explode;

/**
 * RBXP
 *
 * Portable Rubix Object File format (RBXP) is a format designed to reliably store and share serialized PHP objects.
 * Based on PHP's native serialization format, RBXP adds additional layers of compression, tamper protection, and class
 * compatibility detection all in one robust format.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RBXP implements Serializer
{
    /**
     * The identifier or "magic number" of the format.
     *
     * @var string
     */
    protected const IDENTIFIER_STRING = "\241RBXP\r\n\032\n";

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
     * The base serializer.
     *
     * @var \Rubix\ML\Persisters\Serializers\Serializer
     */
    protected $base;

    /**
     * @param string $password
     * @param \Rubix\ML\Persisters\Serializers\Serializer $base
     */
    public function __construct(string $password = '', ?Serializer $base = null)
    {
        $this->password = $password;
        $this->base = $base ?? new Gzip(9, new Native());
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

        $hash = hash_hmac(self::HASHING_FUNCTION, $encoding, $this->password);

        $header = JSON::encode([
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
                'length' => $encoding->bytes(),
            ],
        ]);

        $hash = hash_hmac(self::HASHING_FUNCTION, $header, $this->password);

        $hmac = self::HASHING_FUNCTION . ':' . $hash;

        $data = self::IDENTIFIER_STRING;
        $data .= $hmac . self::EOL;
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
            throw new RuntimeException('Header verification failed.');
        }

        $header = JSON::decode($header);

        switch ($header['data']['hmac']['type']) {
            case 'sha256':
                $hash = hash_hmac('sha256', $payload, $this->password);

                break;

            default:
                throw new RuntimeException('Invalid HMAC hash type.');
        }

        if (!hash_equals($hash, $header['data']['hmac']['token'])) {
            throw new RuntimeException('Data verification failed.');
        }

        $encoding = new Encoding($payload);

        if ($encoding->bytes() !== $header['data']['length']) {
            throw new RuntimeException('Data is corrupted.');
        }

        $persistable = $this->base->unserialize($encoding);

        if (get_class($persistable) !== $header['class']['name']) {
            throw new RuntimeException('Class name mismatch.');
        }

        if ($persistable->revision() !== $header['class']['revision']) {
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
        return 'RBXP';
    }
}
