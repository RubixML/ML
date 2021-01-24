<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Other\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;

use function extension_loaded;
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
 * providing verifiability like the standard RBX format, RBXE encrypts the file data so that it cannot be read without the
 * password.
 *
 * > **Note:** Requires the PHP Open SSL extension to be installed.
 *
 * References:
 * [1] H. Krawczyk et al. (1997). HMAC: Keyed-Hashing for Message Authentication.
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
     * The method used to encrypt the data.
     *
     * @var string
     */
    protected const ENCRYPTION_METHOD = 'aes256';

    /**
     * The number of bytes in the initialization vector.
     *
     * @var int
     */
    protected const INITIALIZATION_BYTES = 16;

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
    public function __construct(string $password = '', ?Serializer $base = null)
    {
        if (!extension_loaded('openssl')) {
            throw new RuntimeException('Open SSL extension is not loaded, check PHP configuration.');
        }

        $digest = password_hash($password, self::DIGEST_HASH_TYPE, [
            'cost' => self::DIGEST_WORK_FACTOR,
        ]);

        if ($digest == false) {
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
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Encoding
     */
    public function serialize(Persistable $persistable) : Encoding
    {
        $iv = random_bytes(self::INITIALIZATION_BYTES);

        $encoding = $this->base->serialize($persistable);

        $encrypted = openssl_encrypt($encoding, self::ENCRYPTION_METHOD, $this->digest, OPENSSL_RAW_DATA, $iv);

        if ($encrypted === false) {
            throw new RuntimeException('Data could not be encrypted.');
        }

        $encoding = new Encoding($encrypted);

        $header = JSON::encode([
            'version' => self::VERSION,
            'class' => [
                'name' => get_class($persistable),
                'revision' => $persistable->revision(),
            ],
            'data' => [
                'encryption' => [
                    'method' => self::ENCRYPTION_METHOD,
                    'iv' => base64_encode($iv),
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

        $iv = base64_decode($header['data']['encryption']['iv']);

        switch ($header['data']['encryption']['method']) {
            case 'aes256':
                $decrypted = openssl_decrypt($payload, 'aes256', $this->digest, OPENSSL_RAW_DATA | OPENSSL_ZERO_PADDING, $iv);

                break;

            default:
                throw new RuntimeException('Invalid encryption method.');
        }

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
        return 'RBXE';
    }
}
