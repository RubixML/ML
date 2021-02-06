<?php

namespace Rubix\ML\Tests\Persisters\Serializers;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Persisters\Serializers\Igbinary;
use Rubix\ML\Persisters\Serializers\Serializer;
use PHPUnit\Framework\TestCase;

/**
 * @group Serializers
 * @requires extension igbinary
 * @covers \Rubix\ML\Persisters\Serializers\Igbinary
 */
class IgbinaryTest extends TestCase
{
    /**
     * @var \Rubix\ML\Persistable
     */
    protected $persistable;

    /**
     * @var \Rubix\ML\Persisters\Serializers\Igbinary
     */
    protected $serializer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->persistable = new DummyClassifier();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->expectDeprecation();

        $this->serializer = new Igbinary();

        $this->assertInstanceOf(Igbinary::class, $this->serializer);
        $this->assertInstanceOf(Serializer::class, $this->serializer);
    }

    /**
     * @test
     */
    public function serializeUnserialize() : void
    {
        $this->expectDeprecation();

        $this->serializer = new Igbinary();

        $data = $this->serializer->serialize($this->persistable);

        $this->assertInstanceOf(Encoding::class, $data);

        $persistable = $this->serializer->unserialize($data);

        $this->assertInstanceOf(DummyClassifier::class, $persistable);
        $this->assertInstanceOf(Persistable::class, $persistable);
    }
}
