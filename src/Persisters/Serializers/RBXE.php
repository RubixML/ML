<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Other\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;

use function extension_loaded;
use function password_hash;
use function strlen;
use function strpos;
use function substr;
use function get_class;
use function hash_hmac;
use function hash_equals;
use function openssl_encrypt;
use function openssl_decrypt;
use function base64_encode;
use function base64_decode;
use function random_bytes;
use function array_pad;
use function explode;

/**
 * RBXE
 *
 * Encrypted Rubix Object File format (RBXE) is a format to securely store and share serialized PHP objects. In addition to
 * ensuring data integrity like RBX format, RBXE also adds layers of security such as tamper protection and data encryption
 * while being resilient to brute-force and evasive to timing attacks.
 *
 * > **Note:** Requires the PHP Open SSL extension to be installed.
 *
 * References:
 * [1] H. Krawczyk et al. (1997). HMAC: Keyed-Hashing for Message Authentication.
 * [2] M. Bellare et al. (2007). Authenticated Encryption: Relations among notions and analysis of the generic composition
 * paradigm.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RBXE implements Serializer
{
    /**
     * The identifier or "magic number" of the format.
     *
     * @var string
     */
    protected const IDENTIFIER_STRING = "\241RBXE\r\n\032\n";

    /**
     * The current version of the format.
     *
     * @var int
     */
    protected const VERSION = 1;

    /**
     * The hashing function used to generate the password digest.
     *
     * @var string
     */
    protected const DIGEST_HASH_TYPE = PASSWORD_BCRYPT;

    /**
     * The work factor of the bcrypt password hashing algorithm.
     *
     * @var int
     */
    protected const DIGEST_WORK_FACTOR = 10;

    /**
     * The hashing function used to generate HMACs.
     *
     * @var string
     */
    protected const HMAC_HASH_TYPE = 'sha256';

    /**
     * The method used to encrypt the data.
     *
     * @var string
     */
    protected const ENCRYPTION_METHOD = 'aes-256-cbc';

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
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function __construct(string $password, ?Serializer $base = null)
    {
        if (!extension_loaded('openssl')) {
            throw new RuntimeException('Open SSL extension is not loaded, check PHP configuration.');
        }

        $digest = password_hash($password, self::DIGEST_HASH_TYPE, [
            'cost' => self::DIGEST_WORK_FACTOR,
        ]);

        if (!$digest) {
            throw new RuntimeException('Could not create digest from password.');
        }

        $this->digest = $digest;
        $this->base = $base ?? new Gzip(9);
    }

    /**
     * Serialize a persistable object and return the data.
     *
     * @internal
     *
     * @param \Rubix\ML\Persistable $persistable
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Encoding
     */
    public function serialize(Persistable $persistable) : Encoding
    {
        $encoding = $this->base->serialize($persistable);

        $iv = random_bytes(openssl_cipher_iv_length(self::ENCRYPTION_METHOD));

        $encrypted = openssl_encrypt($encoding, self::ENCRYPTION_METHOD, $this->digest, OPENSSL_RAW_DATA, $iv);

        if ($encrypted === false) {
            throw new RuntimeException('Data could not be encrypted.');
        }

        $hash = hash_hmac(self::HMAC_HASH_TYPE, $encrypted, $this->digest);

        $header = JSON::encode([
            'class' => [
                'name' => get_class($persistable),
                'revision' => $persistable->revision(),
            ],
            'data' => [
                'hmac' => [
                    'type' => self::HMAC_HASH_TYPE,
                    'token' => $hash,
                ],
                'encryption' => [
                    'method' => self::ENCRYPTION_METHOD,
                    'iv' => base64_encode($iv),
                ],
                'length' => strlen($encrypted),
            ],
        ]);

        $hash = hash_hmac(self::HMAC_HASH_TYPE, $header, $this->digest);

        $hmac = self::HMAC_HASH_TYPE . ':' . $hash;

        $data = self::IDENTIFIER_STRING;
        $data .= self::VERSION . self::EOL;
        $data .= $hmac . self::EOL;
        $data .= $header . self::EOL;
        $data .= $encrypted;

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

        [$version, $hmac, $header, $payload] = array_pad(explode(self::EOL, $data, 4), 4, null);

        if (!$version or !$hmac or !$header or !$payload) {
            throw new RuntimeException('Invalid message format.');
        }

        [$type, $token] = array_pad(explode(':', $hmac, 2), 2, null);

        $hash = hash_hmac($type, $header, $this->digest);

        if (!hash_equals($hash, $token)) {
            throw new RuntimeException('Header HMAC verification failed.');
        }

        $header = JSON::decode($header);

        if (strlen($payload) !== $header['data']['length']) {
            throw new RuntimeException('Data is corrupted.');
        }

        $hash = hash_hmac($header['data']['hmac']['type'], $payload, $this->digest);

        if (!hash_equals($hash, $header['data']['hmac']['token'])) {
            throw new RuntimeException('Data HMAC verification failed.');
        }

        $method = $header['data']['encryption']['method'];

        $iv = base64_decode($header['data']['encryption']['iv']);

        $decrypted = openssl_decrypt($payload, $method, $this->digest, OPENSSL_RAW_DATA, $iv);

        if ($decrypted === false) {
            throw new RuntimeException('Data could not be decrypted.');
        }

        $persistable = $this->base->unserialize(new Encoding($decrypted));

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
        return "RBXE (base: {$this->base})";
    }
}
