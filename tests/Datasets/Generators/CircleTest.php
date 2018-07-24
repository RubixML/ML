<?php

namespace Rubix\Tests\Other\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Generators\Generator;
use PHPUnit\Framework\TestCase;

class CircleTest extends TestCase
{
    protected $generator;

    public function setUp()
    {
        $this->generator = new Circle([0, 0], 1.0, 1.0);
    }

    public function test_build_generator()
    {
        $this->assertInstanceOf(Circle::class, $this->generator);
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
