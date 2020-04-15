<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\VarianceThresholdFilter;
use PHPUnit\Framework\TestCase;
use RuntimeException;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\VarianceThresholdFilter
 */
class VarianceThresholdFilterTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Blob
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Transformers\VarianceThresholdFilter
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Blob([0.0, 0.0, 0.0], [1.0, 5.0, 0.001]);

        $this->transformer = new VarianceThresholdFilter(2);
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(VarianceThresholdFilter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }
    
    /**
     * @test
     */
    public function fitTransform() : void
    {
        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(2, $sample);
        
        $this->assertEqualsWithDelta(0, $sample[0], 3);
        $this->assertEqualsWithDelta(0, $sample[1], 15);
    }
    
    /**
     * @test
     */
    public function transformUfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
