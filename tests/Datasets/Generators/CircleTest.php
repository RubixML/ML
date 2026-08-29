<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Datasets\Generators;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Circle;
use PHPUnit\Framework\TestCase;

#[Group('Generators')]
#[CoversClass(Circle::class)]
class CircleTest extends TestCase
{
    protected const int DATASET_SIZE = 30;

    protected Circle $generator;

    protected function setUp() : void
    {
        $this->generator = new Circle(x: 5.0, y: 5.0, scale: 10.0, noise: 0.1);
    }

    public function testDimensions() : void
    {
        $this->assertEquals(2, $this->generator->dimensions());
    }

    public function testGenerate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(self::DATASET_SIZE, $dataset);
    }
}
