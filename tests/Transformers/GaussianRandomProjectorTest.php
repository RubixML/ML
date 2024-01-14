<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\GaussianRandomProjector;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\GaussianRandomProjector
 */
class GaussianRandomProjectorTest extends TestCase
{
    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @var Blob
     */
    protected $generator;

    /**
     * @var GaussianRandomProjector
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Blob(array_fill(0, 20, 0.0), 3.0);

        $this->transformer = new GaussianRandomProjector(5);

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(GaussianRandomProjector::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    /**
     * @test
     * @dataProvider minDimensionsProvider
     *
     * @param int $n
     * @param float $maxDistortion
     * @param int $expected
     */
    public function minDimensions(int $n, float $maxDistortion, int $expected) : void
    {
        $this->assertEqualsWithDelta($expected, GaussianRandomProjector::minDimensions($n, $maxDistortion), 1e-8);
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function minDimensionsProvider() : Generator
    {
        yield [10, 0.1, 1974];

        yield [100, 0.1, 3947];

        yield [1000, 0.1, 5921];

        yield [10000, 0.1, 7895];

        yield [100000, 0.1, 9868];

        yield [1000000, 0.1, 11842];

        yield [10000, 0.01, 741772];

        yield [10000, 0.3, 1023];

        yield [10000, 0.5, 442];

        yield [10000, 0.99, 221];
    }

    /**
     * @test
     */
    public function fitTransform() : void
    {
        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(5, $sample);
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
