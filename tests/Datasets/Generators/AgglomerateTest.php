<?php

namespace Rubix\ML\Tests\Datasets\Generators;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Generator;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

/**
 * @group Generators
 * @covers \Rubix\ML\Datasets\Generators\Agglomerate
 */
class AgglomerateTest extends TestCase
{
    protected const DATASET_SIZE = 30;

    /**
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'one' => new Blob([-5.0, 3.0], 0.2),
            'two' => new Blob([5.0, -3.0], 0.2),
        ], [1, 0.5]);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Agglomerate::class, $this->generator);
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
        $this->assertEquals(['one', 'two'], $dataset->possibleOutcomes());
    }
}
