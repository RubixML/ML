<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\RecursiveFeatureEliminator;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\RuntimeException;

class RecursiveFeatureEliminatorTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Agglomerate
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Transformers\RecursiveFeatureEliminator
     */
    protected $transformer;

    /**
     * @before
     */
    public function setUp() : void
    {
        $this->generator = new Agglomerate([
            'male' => new Blob([69.2, 195.7, 40.0], [1.0, 3.0, 0.3]),
            'female' => new Blob([63.7, 168.5, 38.1], [0.8, 2.5, 0.4]),
        ], [0.45, 0.55]);

        $this->transformer = new RecursiveFeatureEliminator(1, 1, 0.5);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RecursiveFeatureEliminator::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitTransform() : void
    {
        $dataset = $this->generator->generate(100);

        $this->assertEquals(3, $dataset->numColumns());

        $dataset->apply($this->transformer);

        $this->assertEquals(1, $dataset->numColumns());
    }

    /**
     * @test
     */
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
