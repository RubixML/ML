<?php

namespace Rubix\ML\Tests\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\HalfMoon;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

class HalfMoonTest extends TestCase
{
    protected $generator;

    public function setUp()
    {
        $this->generator = new HalfMoon(5., 5., 10., 45., 0.1);
    }

    public function test_build_generator()
    {
        $this->assertInstanceOf(HalfMoon::class, $this->generator);
        $this->assertInstanceOf(Generator::class, $this->generator);

        $this->assertEquals(2, $this->generator->dimensions());
    }

    public function test_generate_dataset()
    {
        $dataset = $this->generator->generate(30);

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(30, $dataset);
    }
}
