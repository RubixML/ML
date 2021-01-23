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
use function hash;
use function json_encode;
use function json_decode;
use function serialize;
use function unserialize;
use function gzdeflate;
use function gzinflate;
use function array_pad;
use function explode;

/**
 * RBX
 *
 * Rubix Object File Format (RBX) is a format designed to reliably store serialized PHP objects. Based on PHP's native serialization
 * format, RBX includes additional features such as compression, tamper protection, and class definition compatibility detection.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RBX implements Serializer
{
    /**
     * The identifier or "magic number" of the file format.
     *
     * @var string
     */
    protected const IDENTIFIER_STRING = "∃RBX\032";

    /**
     * The current version of the file format.
     *
     * @var int
     */
    protected const VERSION = 1;

    /**
     * The function used to generate checksums.
     *
     * @var string
     */
    protected const HASHING_FUNCTION = 'crc32b';

    /**
     * The end of line character.
     *
     * @var string
     */
    protected const EOL = "\n";

    /**
     * The compression level between 0 and 9, 0 meaning no compression.
     *
     * @var int
     */
    protected $level;

    /**
     * @param int $level
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $level = 9)
    {
        if ($level < 0 or $level > 9) {
            throw new InvalidArgumentException('Compression level must'
                . " be between 0 and 9, $level given.");
        }

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

        $body = gzdeflate($body, $this->level);

        if ($body === false) {
            throw new RuntimeException('Error compressing the data.');
        }

        $header = json_encode([
            'version' => self::VERSION,
            'class' => [
                'name' => get_class($persistable),
                'revision' => $persistable->revision(),
            ],
            'data' => [
                'format' => 'native',
                'compression' => 'deflate',
                'checksum' => hash(self::HASHING_FUNCTION, $body),
                'length' => strlen($body),
            ],
        ]) ?: '';

        $checksum = hash(self::HASHING_FUNCTION, $header);

        $data = self::IDENTIFIER_STRING . self::EOL;
        $data .= $checksum . self::EOL;
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
        [$id, $checksum, $header, $body] = array_pad(explode(self::EOL, $encoding, 4), 4, null);

        if ($id !== self::IDENTIFIER_STRING) {
            throw new RuntimeException('Unrecognized file format.');
        }

        if (!$checksum or !$header or !$body) {
            throw new RuntimeException('Invalid file format.');
        }

        if (hash(self::HASHING_FUNCTION, $header) !== $checksum) {
            throw new RuntimeException('Header checksum does not match.');
        }

        $header = json_decode($header);

        if (strlen($body) !== $header->data->length) {
            throw new RuntimeException('Data has been corrupted.');
        }

        if (hash(self::HASHING_FUNCTION, $body) !== $header->data->checksum) {
            throw new RuntimeException('Data checksum does not match.');
        }

        switch ($header->data->compression) {
            case 'deflate':
                $body = gzinflate($body);

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
            throw new RuntimeException('Missing class definition'
                . ' for unserialized object.');
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
