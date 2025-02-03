<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Datasets\Generators;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Hyperplane;
use PHPUnit\Framework\TestCase;

#[Group('Generators')]
#[CoversClass(Hyperplane::class)]
class HyperplaneTest extends TestCase
{
    protected Hyperplane $generator;

    protected function setUp() : void
    {
        $this->generator = new Hyperplane(coefficients: [0.001, -4.0, 12], intercept: 5.0);
    }

    public function testDimensions() : void
    {
        $this->assertEquals(3, $this->generator->dimensions());
    }

    public function testGenerate() : void
    {
        $dataset = $this->generator->generate(30);

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(30, $dataset);
    }
}
