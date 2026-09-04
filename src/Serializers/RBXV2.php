<?php

namespace Rubix\ML\Serializers;

use Rubix\ML\Set;
use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Specifications\RBXV2HeaderSchemaIsValid;
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
 * Rubix Object File format v2 (RBX) is an improvement upon the RBX V1 format that adds
 * additional layers of security and integrity checks to ensure that serialized objects
 * are not tampered with or corrupted during storage or transmission.
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
    protected const string IDENTIFIER_STRING = "\241RBX\r\n\032\n";

    /**
     * The current version of the format.
     *
     * @var string
     */
    protected const string VERSION = '2';

    /**
     * The hashing function used to generate header checksums.
     *
     * @var string
     */
    protected const string HEADER_CHECKSUM_TYPE = 'sha256';

    /**
     * The hashing function used to generate payload checksums.
     *
     * @var string
     */
    protected const string PAYLOAD_CHECKSUM_TYPE = 'crc32b';

    /**
     * The end of line character.
     *
     * @var string
     */
    protected const string EOL = "\n";

    /**
     * Collect the set of class names present in an object's property graph.
     *
     * @internal
     *
     * @param object $root
     * @return Set
     */
    protected static function collectClassNames(object $root) : Set
    {
        $stack = [$root];

        $seen = new SplObjectStorage();
        $classNames = new Set();

        while ($stack) {
            $value = array_pop($stack);

            if (is_array($value)) {
                foreach ($value as $element) {
                    $stack[] = $element;
                }

                continue;
            }

            if (!is_object($value)) {
                continue;
            }

            if (isset($seen[$value])) {
                continue;
            }

            $seen[$value] = true;

            $reflector = new ReflectionClass($value);

            $classNames->add($reflector->getName());

            $properties = self::collectProperties($reflector);

            foreach ($properties as $property) {
                if ($property->isInitialized($value)) {
                    $stack[] = $property->getValue($value);
                }
            }
        }

        return $classNames;
    }

    /**
     * Enumerate the properties declared throughout a class's inheritance hierarchy,
     * including private members of its parent classes.
     *
     * @internal
     *
     * @param ReflectionClass<object> $reflector
     * @return \ReflectionProperty[]
     */
    protected static function collectProperties(ReflectionClass $reflector) : array
    {
        $properties = [];

        $class = $reflector;

        while ($class) {
            foreach ($class->getProperties() as $property) {
                if ($property->getDeclaringClass()->getName() === $class->getName()) {
                    $properties[] = $property;
                }
            }

            $class = $class->getParentClass();
        }

        return $properties;
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

        $classSet = self::collectClassNames($persistable);

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

        if ($version !== self::VERSION) {
            throw new RuntimeException('Incompatible version format, use the'
                . " RBX V{$version} serializer instead.");
        }

        [$type, $hash] = array_pad(explode(':', $checksum, 2), 2, null);

        if (empty($type) or empty($hash)) {
            throw new RuntimeException('Invalid header digest.');
        }

        if ($type !== self::HEADER_CHECKSUM_TYPE) {
            throw new RuntimeException('Invalid header checksum type.');
        }

        if (hash($type, $header) !== $hash) {
            throw new RuntimeException('Header checksum verification failed.');
        }

        $header = JSON::decode($header);

        RBXV2HeaderSchemaIsValid::with($header)->check();

        $className = $header['class']['name'];
        $allowed = $header['class']['allowed'];
        $revision = $header['class']['revision'];
        $type = $header['data']['checksum']['type'];
        $hash = $header['data']['checksum']['hash'];
        $length = $header['data']['length'];

        if (strlen($payload) !== $length) {
            throw new RuntimeException('Data length does not match header.');
        }

        if ($type !== self::PAYLOAD_CHECKSUM_TYPE) {
            throw new RuntimeException('Invalid data checksum type.');
        }

        if (hash($type, $payload) !== $hash) {
            throw new RuntimeException('Data checksum verification failed.');
        }

        $persistable = unserialize($payload, [
            'allowed_classes' => $allowed,
        ]);

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Missing class for object data.');
        }

        if ($persistable->revision() !== $revision) {
            warn("Class revision mismatch, expected $revision but"
                . " got {$persistable->revision()}. ");
        }

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
