<?php

namespace Rubix\ML\Tests\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

/**
 * @group Generators
 * @covers \Rubix\ML\Datasets\Generators\Circle
 */
class CircleTest extends TestCase
{
    protected const DATASET_SIZE = 30;

    /**
     * @var Circle
     */
    protected $generator;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Circle(5.0, 5.0, 10.0, 0.1);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Circle::class, $this->generator);
        $this->assertInstanceOf(Generator::class, $this->generator);
    }

    /**
     * @test
     */
    public function dimensions() : void
    {
        $this->assertEquals(2, $this->generator->dimensions());
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
