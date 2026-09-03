<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Persisters\Serializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Serializers\RBXV1;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use stdClass;

use function serialize;

#[Group('Serializers')]
#[CoversClass(RBXV1::class)]
class RBXV1Test extends TestCase
{
    protected Persistable $persistable;

    protected RBXV1 $serializer;

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
        $this->serializer = new RBXV1();

        $this->persistable = new AdaBoost();
    }

    #[Test]
    public function serializeDeserialize() : void
    {
        $data = $this->serializer->serialize($this->persistable);

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(AdaBoost::class, $persistable);
    }

    /**
     * @param int|object $obj
     */
    #[DataProvider('deserializeInvalidData')]
    #[Test]
    public function deserializeBadData(mixed $obj) : void
    {
        $data = new Encoding(serialize($obj));

        $this->expectException(RuntimeException::class);

        $this->serializer->deserialize($data);
    }
}
