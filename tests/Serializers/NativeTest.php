<?php

namespace Rubix\ML\Tests\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Serializers\Native;
use Rubix\ML\Serializers\Serializer;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use stdClass;

use function serialize;

/**
 * @group Serializers
 * @covers \Rubix\ML\Serializers\Native
 */
class NativeTest extends TestCase
{
    /**
     * @var \Rubix\ML\Persistable
     */
    protected $persistable;

    /**
     * @var \Rubix\ML\Serializers\Native
     */
    protected $serializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->serializer = new Native();

        $this->persistable = new GaussianNB();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Native::class, $this->serializer);
        $this->assertInstanceOf(Serializer::class, $this->serializer);
    }

    /**
     * @test
     */
    public function serializeDeserialize() : void
    {
        $data = $this->serializer->serialize($this->persistable);

        $this->assertInstanceOf(Encoding::class, $data);

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(GaussianNB::class, $persistable);
        $this->assertInstanceOf(Persistable::class, $persistable);
    }

    /**
     * @return array<mixed>
     */
    public function deserializeInvalidData() : array
    {
        return [
            [3],
            [new stdClass()],
        ];
    }

    /**
     * @test
     *
     * @param mixed $obj
     *
     * @dataProvider deserializeInvalidData
     */
    public function deserializeBadData($obj) : void
    {
        $data = new Encoding(serialize($obj));

        $this->expectException(RuntimeException::class);

        $this->serializer->deserialize($data);
    }
}
