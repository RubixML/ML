<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Persisters\Serializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Serializers\Native;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use stdClass;

use function serialize;

#[Group('Serializers')]
#[CoversClass(Native::class)]
class NativeTest extends TestCase
{
    protected Persistable $persistable;

    protected Native $serializer;

    /**
     * @return array<array<int>|array<object>>
     */
    public static function deserializeInvalidData() : array
    {
        return [
            [3],
            [new stdClass()],
        ];
    }

    protected function setUp() : void
    {
        $this->serializer = new Native();

        $this->persistable = new GaussianNB();
    }

    public function testSerializeDeserialize() : void
    {
        $data = $this->serializer->serialize($this->persistable);

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(GaussianNB::class, $persistable);
    }

    /**
     * @param int|object $obj
     */
    #[DataProvider('deserializeInvalidData')]
    public function testDeserializeBadData(mixed $obj) : void
    {
        $data = new Encoding(serialize($obj));

        $this->expectException(RuntimeException::class);

        $this->serializer->deserialize($data);
    }
}
