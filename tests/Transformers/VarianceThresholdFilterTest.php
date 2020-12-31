<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\VarianceThresholdFilter;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\RuntimeException;

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
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->expectDeprecation();

        $this->transformer = new VarianceThresholdFilter(2);

        $this->assertInstanceOf(VarianceThresholdFilter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitTransform() : void
    {
        $this->expectDeprecation();

        $this->transformer = new VarianceThresholdFilter(2);

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $variances = $this->transformer->variances();

        $this->assertIsArray($variances);
        $this->assertContainsOnly('float', $variances);

        $dataset = $this->generator->generate(3)
            ->apply($this->transformer);

        $this->assertSame(2, $dataset->numColumns());
    }

    /**
     * @test
     */
    public function transformUnfitted() : void
    {
        $this->expectDeprecation();

        $this->transformer = new VarianceThresholdFilter(2);

        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
