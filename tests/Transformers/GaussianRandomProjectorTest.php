<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\GaussianRandomProjector;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Generator;
use Rubix\ML\CrossValidation\Metrics\Trustworthiness;

#[Group('Transformers')]
#[CoversClass(GaussianRandomProjector::class)]
class GaussianRandomProjectorTest extends TestCase
{
    /**
     * The number of samples used for the trustworthiness check.
     */
    protected const int TEST_SIZE = 64;

    /**
     * The minimum trustworthiness score required to pass the test.
     */
    protected const float MIN_SCORE = 0.6;

    /**
     * Constant used to seed the random number generator.
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
    #[Test]
    public function minDimensions(int $n, float $maxDistortion, int $expected) : void
    {
        $this->assertEqualsWithDelta($expected, GaussianRandomProjector::minDimensions($n, $maxDistortion), 1e-8);
    }

    #[Test]
    public function fitTransform() : void
    {
        $this->assertCount(20, $this->generator->generate(1)->sample(0));

        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $dataset = $this->generator->generate(30);

        $originals = $dataset->samples();

        $dataset->apply($this->transformer);

        $projected = $dataset->samples();

        $this->assertCount(5, $projected[0]);

        $meanFactor = 0.0;

        foreach ($originals as $idx => $original) {
            $denominator = $this->squaredNorm($original);

            $meanFactor += $this->squaredNorm($projected[$idx]) / $denominator;
        }

        $meanFactor /= count($originals);

        $this->assertGreaterThan(0.7, $meanFactor, 'Projector does not preserve magnitude (too small).');
        $this->assertLessThan(1.3, $meanFactor, 'Projector does not preserve magnitude (too large).');
    }

    #[Test]
    public function trustworthiness() : void
    {
        srand(self::RANDOM_SEED);

        $generator = new Agglomerate([
            'red' => new Blob([255, 32, 0], 30.0),
            'green' => new Blob([0, 128, 0], 10.0),
            'blue' => new Blob([0, 32, 255], 20.0),
        ], [2, 3, 4]);

        $dataset = $generator->generate(self::TEST_SIZE);

        $original = clone $dataset;

        $reduction = new GaussianRandomProjector(1);

        $reduction->fit($dataset);

        $transformed = $dataset->samples();

        $reduction->transform($transformed);

        $score = (new Trustworthiness())->score($original->samples(), $transformed);

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }

    /**
     * @param array<float> $x
     * @return float
     */
    protected function squaredNorm(array $x) : float
    {
        $sum = 0.0;

        foreach ($x as $value) {
            $sum += $value ** 2;
        }

        return $sum;
    }
}
