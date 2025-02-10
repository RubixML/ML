<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\GaussianRandomProjector;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Transformers')]
#[CoversClass(GaussianRandomProjector::class)]
class GaussianRandomProjectorTest extends TestCase
{
    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Blob $generator;

    protected GaussianRandomProjector $transformer;

    public static function minDimensionsProvider() : Generator
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

    protected function setUp() : void
    {
        $this->generator = new Blob(
            center: array_fill(start_index: 0, count: 20, value: 0.0),
            stdDev: 3.0
        );

        $this->transformer = new GaussianRandomProjector(5);

        srand(self::RANDOM_SEED);
    }

    /**
     * @param int $n
     * @param float $maxDistortion
     * @param int $expected
     */
    #[DataProvider('minDimensionsProvider')]
    public function testMinDimensions(int $n, float $maxDistortion, int $expected) : void
    {
        $this->assertEqualsWithDelta($expected, GaussianRandomProjector::minDimensions($n, $maxDistortion), 1e-8);
    }

    public function testFitTransform() : void
    {
        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(5, $sample);
    }

    public function testTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
