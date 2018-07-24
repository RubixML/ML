<?php

namespace Rubix\Tests\Other\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\HalfMoon;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

class HalfMoonTest extends TestCase
{
    protected $generator;

    public function setUp()
    {
        $this->generator = new HalfMoon([0, 0], 1.0, 90.0, 0.1);
    }

    public function test_build_generator()
    {
        $this->assertInstanceOf(HalfMoon::class, $this->generator);
        $this->assertInstanceOf(Generator::class, $this->generator);
    }

    public function test_generate_dataset()
    {
        $dataset = $this->generator->generate(10);

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(10, $dataset);
    }
}
