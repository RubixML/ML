<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Serializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\JSON;
use Rubix\ML\Serializers\RBXV2;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function serialize;
use function hash;

#[Group('Serializers')]
#[CoversClass(RBXV2::class)]
class RBXV2Test extends TestCase
{
    protected const IDENTIFIER = "\241RBX\r\n\032\n";

    protected RBXV2 $serializer;

    protected function setUp() : void
    {
        $this->serializer = new RBXV2();
    }

    #[Test]
    public function serializeDeserialize() : void
    {
        $data = $this->serializer->serialize(new AdaBoost());

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(AdaBoost::class, $persistable);
    }

    #[Test]
    public function preservesNestedObjects() : void
    {
        $estimator = new KNearestNeighbors(3, false, new Manhattan());
        $data = $this->serializer->serialize($estimator);

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(KNearestNeighbors::class, $persistable);
        $this->assertInstanceOf(Manhattan::class, $persistable->params()['kernel']);
    }

    #[Test]
    public function collectsParentPrivateProperty() : void
    {
        $carrier = new ChildCarrier(new ParentPrivateGadget());
        $data = $this->serializer->serialize($carrier);

        $this->assertContains(ParentPrivateGadget::class, $this->allowedClasses($data));

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(ChildCarrier::class, $persistable);
        $this->assertInstanceOf(ParentPrivateGadget::class, $persistable->gadget());
    }

    #[Test]
    public function collectsNestedArrayObjects() : void
    {
        $carrier = new ArrayCarrier(['one' => ['two' => ['three' => new ArrayLeaf()]]]);
        $data = $this->serializer->serialize($carrier);

        $this->assertContains(ArrayLeaf::class, $this->allowedClasses($data));

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(ArrayCarrier::class, $persistable);
        $this->assertInstanceOf(ArrayLeaf::class, $persistable->deep['one']['two']['three']);
    }

    #[Test]
    public function rejectsBadMagic() : void
    {
        $data = $this->serializer->serialize(new AdaBoost());

        $this->expectException(RuntimeException::class);

        $this->serializer->deserialize(new Encoding("\x00" . substr((string) $data, 1)));
    }

    #[Test]
    public function rejectsUnsupportedVersion() : void
    {
        $persistable = new AdaBoost();
        $payload = serialize($persistable);

        $this->expectException(RuntimeException::class);

        $this->serializer->deserialize(new Encoding(
            $this->makeFile('9', AdaBoost::class, $persistable->revision(), [AdaBoost::class], $payload)
        ));
    }

    #[Test]
    public function rejectsUnsupportedAlgoName() : void
    {
        $persistable = new AdaBoost();
        $payload = serialize($persistable);
        $revision = $persistable->revision();
        $set = [AdaBoost::class];

        $header = $this->makeHeader(AdaBoost::class, $revision, $set, $payload);

        $file = self::IDENTIFIER . "2\n" . 'sha1:' . hash('sha1', $header) . "\n" . $header . "\n" . $payload;

        $this->expectException(RuntimeException::class);

        $this->serializer->deserialize(new Encoding($file));
    }

    #[Test]
    public function rejectsCorruptedHeaderChecksum() : void
    {
        $persistable = new AdaBoost();
        $payload = serialize($persistable);
        $revision = $persistable->revision();
        $set = [AdaBoost::class];

        $header = $this->makeHeader(AdaBoost::class, $revision, $set, $payload);

        $good = hash('sha256', $header);
        $bad = substr($good, 0, -1) . ($good[-1] === '0' ? '1' : '0');

        $file = self::IDENTIFIER . "2\n" . "sha256:$bad\n" . $header . "\n" . $payload;

        $this->expectException(RuntimeException::class);

        $this->serializer->deserialize(new Encoding($file));
    }

    #[Test]
    public function rejectsCorruptedPayload() : void
    {
        $persistable = new AdaBoost();
        $goodPayload = serialize($persistable);
        $badPayload = 'P' . substr($goodPayload, 1);
        $revision = $persistable->revision();
        $set = [AdaBoost::class];

        $header = $this->makeHeader(AdaBoost::class, $revision, $set, $goodPayload);
        $file = self::IDENTIFIER . "2\n" . 'sha256:' . hash('sha256', $header) . "\n" . $header . "\n" . $badPayload;

        $this->expectException(RuntimeException::class);

        $this->serializer->deserialize(new Encoding($file));
    }

    #[Test]
    public function rejectsClassMismatch() : void
    {
        $persistable = new GaussianNB();
        $payload = serialize($persistable);
        $revision = $persistable->revision();

        $file = $this->makeFile(
            '2',
            AdaBoost::class,
            $revision,
            [AdaBoost::class, GaussianNB::class],
            $payload
        );

        $this->expectException(RuntimeException::class);

        $this->serializer->deserialize(new Encoding($file));
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $this->assertIsString((string) $this->serializer);
    }

    protected function makeHeader(string $name, string $revision, array $set, string $payload, string $library = '3') : string
    {
        return JSON::encode([
            'library' => ['version' => $library],
            'class' => [
                'name' => $name,
                'revision' => $revision,
                'allowed' => $set,
            ],
'data' => [
    'checksum' => ['type' => 'crc32b', 'hash' => hash('crc32b', $payload)],
    'length' => strlen($payload),
],
        ]);
    }

    protected function makeFile(
        string $version,
        string $name,
        string $revision,
        array $set,
        string $payload,
        string $library = '3'
    ) : string {
        $header = $this->makeHeader($name, $revision, $set, $payload, $library);
        $checksum = 'sha256:' . hash('sha256', $header);

        return self::IDENTIFIER . $version . "\n" . $checksum . "\n" . $header . "\n" . $payload;
    }

    protected function allowedClasses(Encoding $data) : array
    {
        $body = substr((string) $data, strlen(self::IDENTIFIER));

        [, , $header] = array_pad(explode("\n", $body, 4), 4, null);

        return JSON::decode($header)['class']['allowed'];
    }
}

class ArrayLeaf
{
}

class ArrayCarrier implements Persistable
{
    public array $deep;

    public function __construct(array $deep)
    {
        $this->deep = $deep;
    }

    public function revision() : string
    {
        return 'array-carrier-rev';
    }
}

class PrivateHolder implements Persistable
{
    private $inner;

    public function __construct(object $inner)
    {
        $this->inner = $inner;
    }

    public function inner() : object
    {
        return $this->inner;
    }

    public function revision() : string
    {
        return 'private-holder-rev';
    }
}

class ParentCarrier extends PrivateHolder
{
}

class ChildCarrier extends ParentCarrier
{
    public function gadget() : object
    {
        return $this->inner();
    }
}

class ParentPrivateGadget
{
}
