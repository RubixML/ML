<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
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
