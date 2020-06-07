<?php

namespace Rubix\ML\Tests\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\SwissRoll;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

/**
 * @group Generators
 * @covers \Rubix\ML\Datasets\Generators\SwissRoll
 */
class SwissRollTest extends TestCase
{
    protected const DATASET_SIZE = 30;

    /**
     * @var \Rubix\ML\Datasets\Generators\SwissRoll
     */
    protected $generator;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new SwissRoll(0.0, 0.0, 0.0, 1.0, 12.0, 0.3);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(SwissRoll::class, $this->generator);
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

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(self::DATASET_SIZE, $dataset);
    }
}
