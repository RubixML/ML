<?php

namespace Rubix\ML\Tests\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\SwissRoll;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

class SwissRollTest extends TestCase
{
    protected $generator;

    public function setUp()
    {
        $this->generator = new SwissRoll(0., 0., 0., 1., 12., 0.3);
    }

    public function test_build_generator()
    {
        $this->assertInstanceOf(SwissRoll::class, $this->generator);
        $this->assertInstanceOf(Generator::class, $this->generator);

        $this->assertEquals(3, $this->generator->dimensions());
    }

    public function test_generate_dataset()
    {
        $dataset = $this->generator->generate(30);

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(30, $dataset);
    }
}
