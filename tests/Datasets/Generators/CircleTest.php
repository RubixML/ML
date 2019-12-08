<?php

namespace Rubix\ML\Tests\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

class CircleTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Circle
     */
    protected $generator;

    public function setUp() : void
    {
        $this->generator = new Circle(5., 5., 10.0, 0.1);
    }

    public function test_build_generator() : void
    {
        $this->assertInstanceOf(Circle::class, $this->generator);
        $this->assertInstanceOf(Generator::class, $this->generator);

        $this->assertEquals(2, $this->generator->dimensions());
    }

    public function test_generate_dataset() : void
    {
        $dataset = $this->generator->generate(30);

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(30, $dataset);
    }
}
