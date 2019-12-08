<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class PrincipalComponentAnalysisTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Generator
     */
    protected $generator;
    
    /**
     * @var \Rubix\ML\Transformers\PrincipalComponentAnalysis
     */
    protected $transformer;

    public function setUp() : void
    {
        $this->generator = new Blob([0., 3000., -6., 25], [1., 30., 0.001, 10.]);

        $this->transformer = new PrincipalComponentAnalysis(2);
    }

    public function test_build_transformer() : void
    {
        $this->assertInstanceOf(PrincipalComponentAnalysis::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    public function test_fit_transform() : void
    {
        $this->assertEquals(4, $this->generator->dimensions());

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(2, $sample);
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
