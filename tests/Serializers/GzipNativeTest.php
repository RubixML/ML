<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Serializers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\IgnoreDeprecations;
use Rubix\ML\Persistable;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Serializers\GzipNative;
use PHPUnit\Framework\TestCase;

#[Group('Serializers')]
#[CoversClass(GzipNative::class)]
#[IgnoreDeprecations]
class GzipNativeTest extends TestCase
{
    protected Persistable $persistable;

    protected GzipNative $serializer;

    protected function setUp() : void
    {
        $this->serializer = new GzipNative(6);

        $this->persistable = new GaussianNB();
    }

    #[Test]
    public function serializeDeserialize() : void
    {
        $data = $this->serializer->serialize($this->persistable);

        $persistable = $this->serializer->deserialize($data);

        $this->assertInstanceOf(GaussianNB::class, $persistable);
    }
}
