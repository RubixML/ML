<?php

namespace Rubix\ML\Tests\Transformers;

use ReflectionMethod;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Transformers\TSNE;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Tensor\Matrix;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\TSNE
 */
class TSNETest extends TestCase
{
    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 30;

    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var TSNE
     */
    protected $embedder;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 32, 0], 30.0),
            'green' => new Blob([0, 128, 0], 10.0),
            'blue' => new Blob([0, 32, 255], 20.0),
        ], [2, 3, 4]);

        $this->embedder = new TSNE(1, 10.0, 10, 12.0, 500, 1e-7, 10, new Euclidean());

        $this->embedder->setLogger(new BlackHole());

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(TSNE::class, $this->embedder);
        $this->assertInstanceOf(Verbose::class, $this->embedder);
    }

    /**
     * @test
     */
    public function badNumDimensions() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(0);
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->embedder->compatibility());
    }

    /**
     * @test
     */
    public function gradient() : void
    {
        $p = Matrix::quick([
            [0.0, 0.3, 0.2],
            [0.3, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ]);

        $y = Matrix::quick([
            [1.0],
            [2.0],
            [3.0],
        ]);

        $distances = Matrix::quick([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]);

        $gradient = $this->invokeGradient($this->embedder, $p, $y, $distances);

        $expected = [
            [-0.37],
            [0.0],
            [0.37],
        ];

        foreach ($gradient->asArray() as $i => $row) {
            foreach ($row as $j => $value) {
                $this->assertEqualsWithDelta($expected[$i][$j], $value, 1e-8);
            }
        }

        // Also cover dofs > 1 (e.g. 3D embedding => dofs=2)
        $embedder = new TSNE(3);

        $y3d = Matrix::quick([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]);

        $gradient3d = $this->invokeGradient($embedder, $p, $y3d, $distances)->asArray();

        $this->assertEqualsWithDelta(-0.424792, $gradient3d[0][0], 1e-6);
        $this->assertEqualsWithDelta(0.0, $gradient3d[1][0], 1e-8);
        $this->assertEqualsWithDelta(0.424792, $gradient3d[2][0], 1e-6);
        $this->assertEqualsWithDelta(0.0, $gradient3d[0][1], 1e-8);
        $this->assertEqualsWithDelta(0.0, $gradient3d[0][2], 1e-8);

    /**
     * @test
     */
    public function transform() : void
    {
        $dataset = $this->generator->generate(self::TEST_SIZE);

        $dataset->apply($this->embedder);

        $this->assertCount(self::TEST_SIZE, $dataset);
        $this->assertCount(1, $dataset->sample(0));

        $losses = $this->embedder->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnly('float', $losses);
    }

    /**
     * @param TSNE $embedder
     * @param Matrix $p
     * @param Matrix $y
     * @param Matrix $distances
     * @return Matrix
     */
    private function invokeGradient(TSNE $embedder, Matrix $p, Matrix $y, Matrix $distances) : Matrix
    {
        $method = new ReflectionMethod(TSNE::class, 'gradient');

        $method->setAccessible(true);

        return $method->invokeArgs($embedder, [$p, $y, $distances]);
    }
}
