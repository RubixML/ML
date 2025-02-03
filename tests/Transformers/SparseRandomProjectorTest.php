<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\SparseRandomProjector;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(SparseRandomProjector::class)]
class SparseRandomProjectorTest extends TestCase
{
    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Blob $generator;

    protected SparseRandomProjector $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob(
            center: array_fill(start_index: 0, count: 10, value: 0.0),
            stdDev: 3.0
        );

        $this->transformer = new SparseRandomProjector(dimensions: 4);

        srand(self::RANDOM_SEED);
    }

    public function testFitTransform() : void
    {
        $this->assertCount(10, $this->generator->generate(1)->sample(0));

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $expected = [
            3.8861419746435,
            -17.801078083484,
            0.29819783331323,
            -12.191560356574,
        ];

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(4, $sample);
        $this->assertEqualsWithDelta($expected, $sample, 1e-8);
    }

    public function testTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
