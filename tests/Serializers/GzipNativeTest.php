<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Persisters\Serializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Persistable;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Serializers\GzipNative;
use PHPUnit\Framework\TestCase;

#[Group('Serializers')]
#[CoversClass(GzipNative::class)]
class GzipNativeTest extends TestCase
{
    protected Persistable $persistable;

    protected GzipNative $serializer;

    protected function setUp() : void
    {
        $this->serializer = new GzipNative(6);

        $this->persistable = new GaussianNB();
    }

    public function testSerializeDeserialize() : void
    {
        $data = $this->serializer->serialize($this->persistable);

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(GaussianNB::class, $persistable);
    }
}
