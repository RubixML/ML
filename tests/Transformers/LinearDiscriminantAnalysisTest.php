<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class LinearDiscriminantAnalysisTest extends TestCase
{
    protected $generator;
    
    protected $transformer;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ], [3, 4, 3]);

        $this->transformer = new LinearDiscriminantAnalysis(1);
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(LinearDiscriminantAnalysis::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    public function test_fit_transform()
    {
        $this->assertEquals(3, $this->generator->dimensions());

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(3)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(1, $sample);
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
