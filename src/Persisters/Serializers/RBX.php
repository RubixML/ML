<?php

namespace Rubix\ML\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Exceptions\RuntimeException;
use __PHP_Incomplete_Class;

use function get_class;
use function crc32;
use function json_encode;
use function json_decode;
use function serialize;
use function unserialize;
use function gzencode;
use function gzdecode;

/**
 * RBX
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
     * The default file headers.
     * 
     * @var mixed[]
     */
    protected const HEADER = [
        'version' => self::VERSION,
        'data' => [
            'format' => 'native',
            'compression' => 'gzip',
        ],
    ];

    /**
     * The level of compression applied to the data.
     * 
     * @var int
     */
    protected const COMPRESSION_LEVEL = 9;

    /**
     * The end of line character.
     * 
     * @var string
     */
    protected const EOL = "\n";

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
        $data = self::IDENTIFIER_STRING . self::EOL;

        $header = json_encode(self::HEADER + [
            'class' => [
                'name' => get_class($persistable),
                'revision' => $persistable->revision(),
            ],
        ]);

        $data .= $header . self::EOL;

        $data .= crc32($header) . self::EOL;

        $data .= gzencode(serialize($persistable), self::COMPRESSION_LEVEL) . self::EOL;

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
        [$id, $header, $checksum, $data] = array_pad(explode(self::EOL, $encoding, 4), 4, null);

        if ($id !== self::IDENTIFIER_STRING) {
            throw new RuntimeException('Unrecognized file format.');
        }

        if (!$header or !$checksum or !$data) {
            throw new RuntimeException('Incomplete file.');
        }

        if (crc32($header) != $checksum) {
            throw new RuntimeException('Header checksum does not match.');
        }

        $header = json_decode($header);

        switch ($header->data->compression) {
            case 'gzip':
                $data = gzdecode($data);

                break;
        }

        switch ($header->data->format) {
            case 'native':
            default:
                $persistable = unserialize($data);
        }

        if ($persistable === false) {
            throw new RuntimeException('Cannot read encoding, wrong'
                . ' format or corrupted data.');
        }

        if (!is_object($persistable)) {
            throw new RuntimeException('Unserialized encoding must'
                . ' be an object.');
        }

        if ($persistable instanceof __PHP_Incomplete_Class) {
            throw new RuntimeException('Missing class definition'
                . ' for unserialized object.');
        }

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Unserialized object must'
                . ' implement the Persistable interface.');
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
        return 'RBX';
    }
}
