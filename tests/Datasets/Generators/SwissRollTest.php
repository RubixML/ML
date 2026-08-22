<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Datasets\Generators;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\SwissRoll;
use PHPUnit\Framework\TestCase;

#[Group('Generators')]
#[CoversClass(SwissRoll::class)]
class SwissRollTest extends TestCase
{
    protected const int DATASET_SIZE = 30;

    protected SwissRoll $generator;

    protected function setUp() : void
    {
        $this->generator = new SwissRoll(x: 0.0, y: 0.0, z: 0.0, scale: 1.0, depth: 12.0, noise: 0.3);
    }

    public function testDimensions() : void
    {
        $this->assertEquals(3, $this->generator->dimensions());
    }

    public function testGenerate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(self::DATASET_SIZE, $dataset);
    }
}
