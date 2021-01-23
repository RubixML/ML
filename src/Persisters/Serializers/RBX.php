<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use __PHP_Incomplete_Class;

use function strlen;
use function is_object;
use function get_class;
use function hash_hmac;
use function json_encode;
use function json_decode;
use function serialize;
use function unserialize;
use function array_pad;
use function explode;

/**
 * RBX
 *
 * Rubix Object File format (RBX) is a format designed to securely and reliably store and share serialized PHP objects.
 * Based on PHP's native serialization format, RBX adds additional layers of tamper protection, compression, and class
 * definition compatibility detection all in one robust format.
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
    protected const HASHING_FUNCTION = 'sha512';

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
    protected $level;

    /**
     * @param string $password
     * @param int $level
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(string $password = '', int $level = 5)
    {
        if ($level < 0 or $level > 9) {
            throw new InvalidArgumentException('Compression level must'
                . " be between 0 and 9, $level given.");
        }

        $this->password = $password;
        $this->level = $level;
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
        $body = serialize($persistable);

        $body = gzencode($body, $this->level);

        if ($body === false) {
            throw new RuntimeException('Error compressing the data.');
        }

        $hash = hash_hmac(self::HASHING_FUNCTION, $body, $this->password);

        $header = json_encode([
            'version' => self::VERSION,
            'class' => [
                'name' => get_class($persistable),
                'revision' => $persistable->revision(),
            ],
            'data' => [
                'format' => 'native',
                'compression' => 'gzip',
                'hmac' => [
                    'type' => self::HASHING_FUNCTION,
                    'hash' => $hash,
                ],
                'length' => strlen($body),
            ],
        ]) ?: '';

        $hash = hash_hmac(self::HASHING_FUNCTION, $header, $this->password);

        $data = self::IDENTIFIER_STRING;
        $data .= $hash . self::EOL;
        $data .= $header . self::EOL;
        $data .= $body;

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

        [$hmac, $header, $body] = array_pad(explode(self::EOL, $data, 3), 3, null);

        if (!$hmac or !$header or !$body) {
            throw new RuntimeException('Invalid message structure.');
        }

        $hash = hash_hmac(self::HASHING_FUNCTION, $header, $this->password);

        if ($hash !== $hmac) {
            throw new RuntimeException('Header authenticity could not be verified.');
        }

        $header = json_decode($header);

        if (strlen($body) !== $header->data->length) {
            throw new RuntimeException('Data has been corrupted.');
        }

        switch ($header->data->hmac->type) {
            case 'sha512':
                $hash = hash_hmac('sha512', $body, $this->password);

                break;

            default:
                throw new RuntimeException('Invalid HMAC hash type.');
        }

        if ($hash !== $header->data->hmac->hash) {
            throw new RuntimeException('Data authenticity could not be verified.');
        }

        switch ($header->data->compression) {
            case 'gzip':
                $body = gzdecode($body);

                if ($body === false) {
                    throw new RuntimeException('Error decompressing the data.');
                }

                break;

            default:
                throw new RuntimeException('Invalid compression method.');
        }

        switch ($header->data->format) {
            case 'native':
                $persistable = unserialize($body);

                break;

            default:
                throw new RuntimeException('Invalid data format.');
        }

        if (!is_object($persistable)) {
            throw new RuntimeException('Unserialized data must be an object.');
        }

        if ($persistable instanceof __PHP_Incomplete_Class) {
            throw new RuntimeException('Missing class definition for unserialized object.');
        }

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Unserialized object must'
                . ' implement the Persistable interface.');
        }

        if (get_class($persistable) !== $header->class->name) {
            throw new RuntimeException('Header and data classes do not match.');
        }

        if ($persistable->revision() !== $header->class->revision) {
            throw new RuntimeException('Class revision numbers do not match.');
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
        return "RBX (level: {$this->level})";
    }
}
