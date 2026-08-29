<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use ReflectionMethod;
use ReflectionProperty;
use Rubix\ML\DataType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Transformers\TSNE;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Tensor\Matrix;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(TSNE::class)]
class TSNETest extends TestCase
{
    /**
     * The number of samples in the validation set.
     */
    protected const int TEST_SIZE = 30;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected TSNE $embedder;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'red' => new Blob([255, 32, 0], 30.0),
                'green' => new Blob([0, 128, 0], 10.0),
                'blue' => new Blob([0, 32, 255], 20.0),
            ],
            weights: [2, 3, 4]
        );

        $this->embedder = new TSNE(
            dimensions: 1,
            rate: 10.0,
            perplexity: 10,
            exaggeration: 12.0,
            epochs: 500,
            minGradient: 0.0
        );

        $this->embedder->setLogger(new BlackHole());

        srand(self::RANDOM_SEED);
    }

    public function testBadNumDimensions() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(dimensions: 0);
    }

    public function testBadRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(rate: 0.0);
    }

    public function testBadPerplexity() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(perplexity: 0);
    }

    public function testBadExaggeration() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(exaggeration: 0.5);
    }

    public function testBadEpochs() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(epochs: 0);
    }

    public function testBadMinGradient() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TSNE(minGradient: -1.0);
    }

    public function testDefaults() : void
    {
        $tsne = new TSNE();

        $this->assertEquals(
            't-SNE (dimensions: 2, rate: 100, perplexity: 30, exaggeration: 12, epochs: 1000, '
            . 'min gradient: 1.0E-7)',
            (string) $tsne
        );
    }

    public function testCompatibility() : void
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
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
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
    }

    /**
     * @test
     */
    public function gradientWeight() : void
    {
        $embedder = new TSNE(
            dimensions: 3,
            rate: 10.0,
            perplexity: 10,
            exaggeration: 12.0,
            epochs: 500,
            minGradient: 1e-7
        );

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
            [0.0, 1.0, 9.0],
            [1.0, 0.0, 4.0],
            [9.0, 4.0, 0.0],
        ]);

        $gradient = $this->invokeGradient($embedder, $p, $y, $distances);

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

        $pwMethod = new ReflectionMethod(TSNE::class, 'squaredPairwiseDistances');
        $pwMethod->setAccessible(true);
        $distances = $pwMethod->invokeArgs($this->embedder, [$y]);

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
        $embedder = new TSNE(
            dimensions: 1,
            rate: 10.0,
            perplexity: 2,
            exaggeration: 12.0,
            epochs: 500,
            minGradient: 1e-7
        );

        $distances = [
            [0.0, 1.0, 4.0, 9.0],
            [1.0, 0.0, 1.0, 4.0],
            [4.0, 1.0, 0.0, 1.0],
            [9.0, 4.0, 1.0, 0.0],
        ];

        $affinities = $this->invokeAffinities($embedder, $distances);

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
        $this->assertContainsOnlyFloat($losses);
    }

    public function testTransformDefaultDimensions() : void
    {
        srand(self::RANDOM_SEED);

        $dataset = $this->generator->generate(self::TEST_SIZE);

        $embedder = new TSNE();

        $embedder->setLogger(new BlackHole());

        $dataset->apply($embedder);

        $this->assertCount(self::TEST_SIZE, $dataset);
        $this->assertCount(2, $dataset->sample(0));

        $this->assertIsArray($embedder->losses());
    }

    public function testEarlyStopOnMinGradient() : void
    {
        srand(self::RANDOM_SEED);

        $earlyStop = new TSNE(
            dimensions: 1,
            epochs: 200,
            minGradient: 1e9
        );

        $earlyStop->setLogger(new BlackHole());

        $this->generator->generate(self::TEST_SIZE)->apply($earlyStop);

        $this->assertCount(1, $earlyStop->losses());

        $steps = iterator_to_array($earlyStop->steps());

        $this->assertCount(1, $steps);
    }

    public function testRunsAllEpochs() : void
    {
        srand(self::RANDOM_SEED);

        $this->generator->generate(self::TEST_SIZE)->apply($this->embedder);

        $this->assertCount(500, $this->embedder->losses());
    }

    public function testSteps() : void
    {
        srand(self::RANDOM_SEED);

        $this->generator->generate(self::TEST_SIZE)->apply($this->embedder);

        $steps = iterator_to_array($this->embedder->steps());

        $this->assertCount(500, $steps);

        $epoch = 0;

        $losses = [];

        foreach ($steps as $step) {
            $this->assertSame($epoch, $step['epoch']);
            $this->assertIsFloat($step['loss']);
            $this->assertGreaterThanOrEqual(0.0, $step['loss']);
            $losses[] = $step['loss'];
            ++$epoch;
        }

        $this->assertEquals($losses, $this->embedder->losses());
    }

    public function testAffinitiesNormalize() : void
    {
        $distances = [
            [0.0, 1.0, 4.0, 9.0],
            [1.0, 0.0, 1.0, 4.0],
            [4.0, 1.0, 0.0, 1.0],
            [9.0, 4.0, 1.0, 0.0],
        ];

        $affinities = $this->invokeAffinities($this->embedder, $distances);

        $this->assertCount(4, $affinities);

        foreach ($affinities as $i => $row) {
            $this->assertCount(4, $row);
            $this->assertEqualsWithDelta(0.25, array_sum($row), 1e-8);
            $this->assertSame(0.0, $row[$i]);
        }
    }

    public function testGradientShape() : void
    {
        $p = Matrix::quick([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.2, 0.3, 0.5],
        ]);

        $y = Matrix::quick([
            [0.0, 1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ]);

        $distances = Matrix::quick([
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
        ]);

        $rows = $this->invokeGradient($this->embedder, $p, $y, $distances)->asArray();

        $this->assertCount(3, $rows);

        foreach ($rows as $row) {
            $this->assertCount(2, $row);
        }
    }

    public function testGradientNonZero() : void
    {
        $p = Matrix::quick([
            [0.9, 0.05, 0.05],
            [0.5, 0.4, 0.1],
            [0.1, 0.2, 0.7],
        ]);

        $y = Matrix::quick([
            [0.0, 1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ]);

        $distances = Matrix::quick([
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
        ]);

        $this->assertGreaterThan(
            0.0,
            $this->invokeGradient($this->embedder, $p, $y, $distances)->l2Norm()
        );
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
     * @param array<float[]> $distances
     * @return array<float[]>
     */
    private function invokeAffinities(TSNE $embedder, array $distances) : array
    {
        $method = new ReflectionMethod(TSNE::class, 'affinities');

        $method->setAccessible(true);

        return $method->invokeArgs($embedder, [Matrix::quick($distances)]);
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

        $pwMethod = new ReflectionMethod(TSNE::class, 'squaredPairwiseDistances');

        $pwMethod->setAccessible(true);

        $distances = $pwMethod->invokeArgs($this->embedder, [$y]);

        $base = $distances->divide($dofs)
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
