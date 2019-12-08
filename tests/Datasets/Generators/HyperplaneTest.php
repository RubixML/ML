<?php

namespace Rubix\ML\Tests\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

class HyperplaneTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Hyperplane
     */
    protected $generator;

    public function setUp() : void
    {
        $this->generator = new Hyperplane([0, 0], 0.0);
    }

    public function test_build_generator() : void
    {
        $this->assertInstanceOf(Hyperplane::class, $this->generator);
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
