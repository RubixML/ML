<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Datasets\Generators;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

#[Group('Generators')]
#[CoversClass(Blob::class)]
class BlobTest extends TestCase
{
    protected const int DATASET_SIZE = 30;

    protected Blob $generator;

    protected function setUp() : void
    {
        $this->generator = new Blob(center: [0, 0, 0], stdDev: 1.0);
    }

    public function testSimulate() : void
    {
        $dataset = $this->generator->generate(100);

        $generator = Blob::simulate($dataset);

        $this->assertInstanceOf(Blob::class, $generator);
        $this->assertInstanceOf(Generator::class, $generator);
    }

    public function testCenter() : void
    {
        $this->assertEquals([0, 0, 0], $this->generator->center());
    }

    public function testDimensions() : void
    {
        $this->assertEquals(3, $this->generator->dimensions());
    }

    public function testGenerate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(self::DATASET_SIZE, $dataset);
    }
}
