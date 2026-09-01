<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @requires extension tensor
 * @covers \Rubix\ML\Transformers\LinearDiscriminantAnalysis
 */
class LinearDiscriminantAnalysisTest extends TestCase
{
    /**
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var LinearDiscriminantAnalysis
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 30.0),
            'green' => new Blob([0, 128, 0], 10.0),
            'blue' => new Blob([0, 0, 255], 20.0),
        ], [3, 4, 3]);

        $this->transformer = new LinearDiscriminantAnalysis(1);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(LinearDiscriminantAnalysis::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitTransform() : void
    {
        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(3)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(1, $sample);
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
