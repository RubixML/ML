<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\SparseRandomProjector;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\SparseRandomProjector
 */
class SparseRandomProjectorTest extends TestCase
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
     * @var SparseRandomProjector
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Blob(array_fill(0, 10, 0.0), 3.0);

        $this->transformer = new SparseRandomProjector(4);

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(SparseRandomProjector::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    /**
     * @test
     */
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

    /**
     * @test
     */
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
