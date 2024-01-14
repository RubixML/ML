<?php

namespace Rubix\ML\Tests\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Serializers\GzipNative;
use Rubix\ML\Serializers\Serializer;
use PHPUnit\Framework\TestCase;

/**
 * @group Serializers
 * @covers \Rubix\ML\Serializers\Gzip
 */
class GzipNativeTest extends TestCase
{
    /**
     * @var Persistable
     */
    protected $persistable;

    /**
     * @var GzipNative
     */
    protected $serializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->serializer = new GzipNative(6);

        $this->persistable = new GaussianNB();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(GzipNative::class, $this->serializer);
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
}
