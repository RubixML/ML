<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\SparseRandomProjector;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Rubix\ML\CrossValidation\Metrics\Trustworthiness;

#[Group('Transformers')]
#[CoversClass(SparseRandomProjector::class)]
class SparseRandomProjectorTest extends TestCase
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

    #[Test]
    public function fitTransform() : void
    {
        $this->assertCount(10, $this->generator->generate(1)->sample(0));

        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $dataset = $this->generator->generate(30);

        $originals = $dataset->samples();

        $dataset->apply($this->transformer);

        $projected = $dataset->samples();

        $this->assertCount(4, $projected[0]);

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

        $reduction = new SparseRandomProjector(1);

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
