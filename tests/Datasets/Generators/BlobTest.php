<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Datasets\Generators;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
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

    #[Test]
    public function simulate() : void
    {
        $dataset = $this->generator->generate(100);

        $generator = Blob::simulate($dataset);

        $this->assertInstanceOf(Blob::class, $generator);
        $this->assertInstanceOf(Generator::class, $generator);
    }

    #[Test]
    public function center() : void
    {
        $this->assertEquals([0, 0, 0], $this->generator->center());
    }

    #[Test]
    public function dimensions() : void
    {
        $this->assertEquals(3, $this->generator->dimensions());
    }

    #[Test]
    public function generate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(self::DATASET_SIZE, $dataset);
    }
}
