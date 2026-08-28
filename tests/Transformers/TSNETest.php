<?php

namespace Rubix\ML\Tests\Transformers;

use ReflectionMethod;
use ReflectionProperty;
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

        $gradient = $this->invokeGradient($this->embedder, $p, $y, $distances->square());

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
    }

    /**
     * @test
     */
    public function gradientWeight() : void
    {
        $embedder = new TSNE(3, 10.0, 10, 12.0, 500, 1e-7, 10, new Euclidean());

        $p = Matrix::quick([
            [0.0, 0.3, 0.2],
            [0.3, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ]);

        $y = Matrix::quick([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]);

        $distances = Matrix::quick([
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ]);

        $gradient = $this->invokeGradient($embedder, $p, $y, $distances->square());

        $expected = [
            [-0.18091856296078745, 0.0, 0.0],
            [-0.4321223317436502, 0.0, 0.0],
            [0.6130408947044377, 0.0, 0.0],
        ];

        foreach ($gradient->asArray() as $i => $row) {
            foreach ($row as $j => $value) {
                $this->assertEqualsWithDelta($expected[$i][$j], $value, 1e-8);
            }
        }
    }

    /**
     * @test
     */
    public function gradientCorrectness() : void
    {
        $p = Matrix::quick([
            [0.0, 0.4, 0.3, 0.3],
            [0.4, 0.0, 0.3, 0.3],
            [0.3, 0.3, 0.0, 0.4],
            [0.3, 0.3, 0.4, 0.0],
        ]);

        $pTotal = $p->sum()->sum();

        $p = $p->divide($pTotal);

        $y = Matrix::quick([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]);

        $pwMethod = new ReflectionMethod(TSNE::class, 'pairwiseDistances');
        $pwMethod->setAccessible(true);
        $distances = Matrix::quick($pwMethod->invokeArgs($this->embedder, [$y->asArray()]))->square();

        $codeGradient = $this->invokeGradient($this->embedder, $p, $y, $distances);

        $eps = 1e-5;
        $numericalGradient = [];

        for ($i = 0; $i < 4; ++$i) {
            $row = [];

            for ($d = 0; $d < 2; ++$d) {
                $yArray = $y->asArray();

                $yArray[$i][$d] += $eps;
                $yPlus = Matrix::quick($yArray);

                $yArray[$i][$d] -= 2.0 * $eps;
                $yMinus = Matrix::quick($yArray);

                $costPlus = $this->klCost($p, $yPlus);
                $costMinus = $this->klCost($p, $yMinus);

                $row[] = ($costPlus - $costMinus) / (2.0 * $eps);
            }

            $numericalGradient[] = $row;
        }

        $numerical = Matrix::quick($numericalGradient);

        $codeNorm = $codeGradient->l2Norm();
        $diff = $codeGradient->subtract($numerical)->l2Norm();

        $this->assertGreaterThan(0.0, $codeNorm);
        $this->assertLessThan(0.05 * $codeNorm, $diff);
    }

    /**
     * @test
     */
    public function affinities() : void
    {
        $embedder = new TSNE(1, 10.0, 2, 12.0, 500, 1e-7, 10, new Euclidean());

        $distances = [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ];

        $affinities = $this->invokeAffinities($embedder, Matrix::quick($distances)->square())->asArray();

        $this->assertCount(4, $affinities);

        $totalSum = 0.0;

        foreach ($affinities as $i => $row) {
            $this->assertCount(4, $row);
            $this->assertSame(0.0, $row[$i]);
            $totalSum += array_sum($row);
        }

        $this->assertEqualsWithDelta(1.0, $totalSum, 1e-8);

        foreach ($affinities as $i => $row) {
            foreach ($row as $j => $value) {
                $this->assertEqualsWithDelta($value, $affinities[$j][$i], 1e-8);
            }
        }
    }

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

    /**
     * @param TSNE $embedder
     * @param Matrix $distances
     * @return Matrix
     */
    private function invokeAffinities(TSNE $embedder, Matrix $distances) : Matrix
    {
        $method = new ReflectionMethod(TSNE::class, 'affinities');

        $method->setAccessible(true);

        return $method->invokeArgs($embedder, [$distances]);
    }

    /**
     * Compute the KL divergence cost C = Σ_{i≠j} p_{ij} log(p_{ij} / q_{ij}).
     *
     * @param Matrix $p
     * @param Matrix $y
     * @return float
     */
    private function klCost(Matrix $p, Matrix $y) : float
    {
        $prop = new ReflectionProperty(TSNE::class, 'dofs');

        $prop->setAccessible(true);

        $dofs = (int) $prop->getValue($this->embedder);

        $pwMethod = new ReflectionMethod(TSNE::class, 'pairwiseDistances');

        $pwMethod->setAccessible(true);

        $distances = Matrix::quick($pwMethod->invokeArgs($this->embedder, [$y->asArray()]));

        $base = $distances->square()
            ->divide($dofs)
            ->add(1.0);

        $kernel = $base->pow((1.0 + $dofs) / -2.0);

        $norm = $kernel->sum()->sum() - $kernel->diagonalAsVector()->sum();

        $q = $kernel->divide(max($norm, 1e-8));

        $pArray = $p->asArray();
        $qArray = $q->asArray();
        $n = $p->m();
        $cost = 0.0;

        for ($i = 0; $i < $n; ++$i) {
            for ($j = 0; $j < $n; ++$j) {
                if ($i !== $j && $pArray[$i][$j] > 0.0) {
                    $cost += $pArray[$i][$j] * log($pArray[$i][$j] / $qArray[$i][$j]);
                }
            }
        }

        return $cost;
    }
}
