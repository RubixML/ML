<?php

namespace Rubix\ML\Serializers;

use Rubix\ML\Set;
use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;
use SplObjectStorage;
use ReflectionClass;

use function Rubix\ML\warn;
use function is_object;
use function is_array;
use function serialize;
use function unserialize;
use function str_starts_with;
use function strlen;
use function substr;
use function hash;
use function get_class;
use function array_pad;
use function explode;

use const Rubix\ML\VERSION as LIBRARY_VERSION;

/**
 * RBX V2
 *
 * Rubix Object File format v2 (RBX) is a format designed to reliably store and share serialized PHP
 * objects. RBX is built directly on PHP's native serialization format and layers data-integrity
 * checksums, class-compatibility detection, and enhanced security, all in one compact format.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RBXV2 implements Serializer
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
     * The hashing function used to generate header checksums.
     *
     * @var string
     */
    protected const HEADER_CHECKSUM_TYPE = 'sha256';

    /**
     * The hashing function used to generate payload checksums.
     *
     * @var string
     */
    protected const PAYLOAD_CHECKSUM_TYPE = 'crc32b';

    /**
     * The end of line character.
     *
     * @var string
     */
    protected const EOL = "\n";

    /**
     * Collect the set of class names present in an object's property graph.
     *
     * @internal
     *
     * @param object $root
     * @return Set
     */
    protected static function collectClassSet(object $root) : Set
    {
        $stack = [$root];

        $seen = new SplObjectStorage();
        $classNames = new Set();

        while ($stack) {
            $current = array_pop($stack);

            if (isset($seen[$current])) {
                continue;
            }

            $seen[$current] = true;

            $reflector = new ReflectionClass($current);

            $className = $reflector->getName();

            $classNames->add($className);

            $properties = $reflector->getProperties();

            foreach ($properties as $property) {
                if (!$property->isInitialized($current)) {
                    continue;
                }

                $value = $property->getValue($current);

                if (is_object($value)) {
                    $stack[] = $value;

                    continue;
                }

                if (is_array($value)) {
                    foreach ($value as $element) {
                        if (is_object($element)) {
                            $stack[] = $element;
                        }
                    }
                }
            }
        }

        return $classNames;
    }

    /**
     * Serialize a persistable object and return the data.
     *
     * @internal
     *
     * @param Persistable $persistable
     * @return Encoding
     */
    public function serialize(Persistable $persistable) : Encoding
    {
        $className = get_class($persistable);

        $classSet = self::collectClassSet($persistable);

        $payload = serialize($persistable);

        $payloadHash = hash(self::PAYLOAD_CHECKSUM_TYPE, $payload);

        $header = JSON::encode([
            'library' => [
                'version' => LIBRARY_VERSION,
            ],
            'class' => [
                'name' => $className,
                'allowed' => $classSet->toArray(),
                'revision' => $persistable->revision(),
            ],
            'data' => [
                'checksum' => [
                    'type' => self::PAYLOAD_CHECKSUM_TYPE,
                    'hash' => $payloadHash,
                ],
                'length' => strlen($payload),
            ],
        ]);

        $headerHash = hash(self::HEADER_CHECKSUM_TYPE, $header);

        $checksum = self::HEADER_CHECKSUM_TYPE . ':' . $headerHash;

        $data = self::IDENTIFIER_STRING;
        $data .= self::VERSION . self::EOL;
        $data .= $checksum . self::EOL;
        $data .= $header . self::EOL;
        $data .= $payload;

        return new Encoding($data);
    }

    /**
     * Deserialize a persistable object and return it.
     *
     * @internal
     *
     * @param Encoding $encoding
     * @throws RuntimeException
     * @return Persistable
     */
    public function deserialize(Encoding $encoding) : Persistable
    {
        if (!str_starts_with($encoding, self::IDENTIFIER_STRING)) {
            throw new RuntimeException('Unrecognized message format.');
        }

        $data = substr($encoding, strlen(self::IDENTIFIER_STRING));

        [$version, $checksum, $header, $payload] = array_pad(explode(self::EOL, $data, 4), 4, null);

        if (empty($version) or empty($checksum) or empty($header) or empty($payload)) {
            throw new RuntimeException('Invalid message format.');
        }

        if ($version != self::VERSION) {
            throw new RuntimeException('Incompatible version format, use the'
                . " RBX V{$version} serializer instead.");
        }

        [$type, $hash] = array_pad(explode(':', $checksum, 2), 2, null);

        if (empty($type) or empty($hash)) {
            throw new RuntimeException('Invalid header digest.');
        }

        if ($type != self::HEADER_CHECKSUM_TYPE) {
            throw new RuntimeException('Invalid header checksum type.');
        }

        if (hash($type, $header) !== $hash) {
            throw new RuntimeException('Header checksum verification failed.');
        }

        $header = JSON::decode($header);

        $length = $header['data']['length'] ?? null;

        if (strlen($payload) !== $length) {
            throw new RuntimeException('Data length does not match header.');
        }

        $type = $header['data']['checksum']['type'] ?? null;

        if ($type !== self::PAYLOAD_CHECKSUM_TYPE) {
            throw new RuntimeException('Invalid data checksum type.');
        }

        $hash = $header['data']['checksum']['hash'] ?? null;

        if (hash($type, $payload) !== $hash) {
            throw new RuntimeException('Data checksum verification failed.');
        }

        $allowedClasses = $header['class']['allowed'] ?? [];

        $persistable = unserialize($payload, ['allowed_classes' => $allowedClasses]);

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Missing class for object data.');
        }

        $expectedRevision = $header['class']['revision'] ?? null;

        if ($persistable->revision() !== $expectedRevision) {
            warn("Class revision mismatch, expected $expectedRevision but"
                . " got {$persistable->revision()}. ");
        }

        $className = $header['class']['name'] ?? null;

        if (get_class($persistable) !== $className) {
            throw new RuntimeException('Class name mismatch.');
        }

        return $persistable;
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
        return 'RBX V2';
    }
}
