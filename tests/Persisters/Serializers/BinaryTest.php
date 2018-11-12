<?php

namespace Rubix\ML\Tests\Persisters\Serializers;

use Rubix\ML\Persistable;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Persisters\Serializers\Binary;
use Rubix\ML\Persisters\Serializers\Serializer;
use PHPUnit\Framework\TestCase;

class BinaryTest extends TestCase
{
    protected $serializer;

    protected $persistable;

    public function setUp()
    {
        $this->serializer = new Binary();

        $this->persistable = new DummyClassifier();
    }

    public function test_build_serialzer()
    {
        $this->assertInstanceOf(Binary::class, $this->serializer);
        $this->assertInstanceOf(Serializer::class, $this->serializer);
    }

    public function test_serialize_unserialize()
    {
        $data = $this->serializer->serialize($this->persistable);
        
        $this->assertInternalType('string', $data);

        $persistable = $this->serializer->unserialize($data);

        $this->assertInstanceOf(DummyClassifier::class, $persistable);
        $this->assertInstanceOf(Persistable::class, $persistable);
    }
}
