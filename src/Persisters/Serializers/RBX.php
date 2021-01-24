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
use function array_pad;
use function explode;

/**
 * RBX
 *
 * Rubix Object File format (RBX) is a format designed to reliably store and share serialized PHP objects. Based on PHP's native
 * serialization format, RBX adds additional layers of compression, tamper protection, and class compatibility detection all in
 * one robust format. Unlike the encrypted RBXE however, file data can still be read even if the authenticity of it cannot be
 * verified with the password.
 *
 * References:
 * [1] H. Krawczyk et al. (1997). HMAC: Keyed-Hashing for Message Authentication.
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
     * The length of the password digest in bytes.
     *
     * @var int
     */
    protected const DIGEST_LENGTH = 32;

    /**
     * The hashing function used to generate the password digest.
     *
     * @var string
     */
    protected const DIGEST_HASH_TYPE = PASSWORD_BCRYPT;

    /**
     * The work factor of the password hashing algorithm.
     *
     * @var int
     */
    protected const DIGEST_WORK_FACTOR = 10;

    /**
     * The hashing function used to generate the header HMAC.
     *
     * @var string
     */
    protected const HEADER_HASH_TYPE = 'sha256';

    /**
     * The hashing function used to generate the payload HMAC.
     *
     * @var string
     */
    protected const PAYLOAD_HASH_TYPE = 'sha512';

    /**
     * The end of line character.
     *
     * @var string
     */
    protected const EOL = "\n";

    /**
     * The hash of the given password.
     *
     * @var string
     */
    protected $digest;

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
        $digest = password_hash($password, self::DIGEST_HASH_TYPE, [
            'cost' => self::DIGEST_WORK_FACTOR,
        ]);

        if ($digest === false) {
            throw new RuntimeException('Could not create digest from password.');
        }

        $digest = substr($digest, 0, self::DIGEST_LENGTH);

        $this->digest = $digest;
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

        $hash = hash_hmac(self::PAYLOAD_HASH_TYPE, $encoding, $this->digest);

        $header = JSON::encode([
            'version' => self::VERSION,
            'class' => [
                'name' => get_class($persistable),
                'revision' => $persistable->revision(),
            ],
            'data' => [
                'hmac' => [
                    'type' => self::PAYLOAD_HASH_TYPE,
                    'token' => $hash,
                ],
                'length' => $encoding->bytes(),
            ],
        ]);

        $hash = hash_hmac(self::HEADER_HASH_TYPE, $header, $this->digest);

        $hmac = self::HEADER_HASH_TYPE . ':' . $hash;

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

        $hash = hash_hmac($type, $header, $this->digest);

        if (!hash_equals($hash, $token)) {
            throw new RuntimeException('Header verification failed.');
        }

        $header = JSON::decode($header);

        if (strlen($payload) !== $header['data']['length']) {
            throw new RuntimeException('Data is corrupted.');
        }

        switch ($header['data']['hmac']['type']) {
            case 'sha512':
                $hash = hash_hmac('sha512', $payload, $this->digest);

                break;

            default:
                throw new RuntimeException('Invalid HMAC hash type.');
        }

        if (!hash_equals($hash, $header['data']['hmac']['token'])) {
            throw new RuntimeException('Data verification failed.');
        }

        $persistable = $this->base->unserialize(new Encoding($payload));

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
        return 'RBX';
    }
}
