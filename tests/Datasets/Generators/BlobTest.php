<?php

namespace Rubix\ML\Tests\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

/**
 * @group Generators
 * @covers \Rubix\ML\Datasets\Generators\Blob
 */
class BlobTest extends TestCase
{
    protected const DATASET_SIZE = 30;

    /**
     * @var \Rubix\ML\Datasets\Generators\Blob
     */
    protected $generator;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Blob([0, 0, 0], 1.0);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Blob::class, $this->generator);
        $this->assertInstanceOf(Generator::class, $this->generator);
    }

    /**
     * @test
     */
    public function dimensions() : void
    {
        $this->assertEquals(3, $this->generator->dimensions());
    }

    /**
     * @test
     */
    public function generate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(self::DATASET_SIZE, $dataset);
    }
}
