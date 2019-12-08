<?php

namespace Rubix\ML\Tests\Persisters\Serializers;

use Rubix\ML\Persistable;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Persisters\Serializers\Native;
use Rubix\ML\Persisters\Serializers\Serializer;
use PHPUnit\Framework\TestCase;

class NativeTest extends TestCase
{
    /**
     * @var \Rubix\ML\Persistable
     */
    protected $persistable;

    /**
     * @var \Rubix\ML\Persisters\Serializers\Native
     */
    protected $serializer;

    public function setUp() : void
    {
        $this->serializer = new Native();

        $this->persistable = new DummyClassifier();
    }

    public function test_build_serialzer() : void
    {
        $this->assertInstanceOf(Native::class, $this->serializer);
        $this->assertInstanceOf(Serializer::class, $this->serializer);
    }

    public function test_serialize_unserialize() : void
    {
        $data = $this->serializer->serialize($this->persistable);
        
        $this->assertIsString($data);

        $persistable = $this->serializer->unserialize($data);

        $this->assertInstanceOf(DummyClassifier::class, $persistable);
        $this->assertInstanceOf(Persistable::class, $persistable);
    }
}
