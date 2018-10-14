<?php

namespace Rubix\ML\Tests\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Generator;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

class AgglomerateTest extends TestCase
{
    protected $generator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'one' => new Blob([-5.0, 3.0], 0.2),
            'two' => new Blob([5.0, -3.0], 0.2),
        ], [1, 0.5]);
    }

    public function test_build_generator()
    {
        $this->assertInstanceOf(Agglomerate::class, $this->generator);
        $this->assertInstanceOf(Generator::class, $this->generator);
    }

    public function test_generate_dataset()
    {
        $dataset = $this->generator->generate(30);

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertCount(30, $dataset);
        $this->assertEquals(['one', 'two'], $dataset->possibleOutcomes());
    }
}
