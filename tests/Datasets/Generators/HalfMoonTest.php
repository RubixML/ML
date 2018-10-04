<?php

namespace Rubix\ML\Tests\Datasets\Generators;

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
        $this->generator = new HalfMoon(5., 5., 10.0, 50.0, 0.1);
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
