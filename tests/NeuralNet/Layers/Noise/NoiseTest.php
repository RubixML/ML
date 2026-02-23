<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers\Noise;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Noise\Noise;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic\Stochastic;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Noise::class)]
class NoiseTest extends TestCase
{
    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected NDArray $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected Noise $layer;

    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->input = NumPower::array([
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ]);

        $this->prevGrad = new Deferred(fn: function () : NDArray {
            return NumPower::array([
                [0.25, 0.7, 0.1],
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Noise(0.1);
    }

    /**
     * @return array<int, array{0: array<int, array<int, float>>}>
     */
    public static function backProvider() : array
    {
        return [
            [
                [
                    [0.25, 0.7, 0.1],
                    [0.5, 0.2, 0.01],
                    [0.25, 0.1, 0.89],
                ],
            ],
        ];
    }

    /**
     * @return array<int, array{0: array<int, array<int, float>>}>
     */
    public static function inferProvider() : array
    {
        return [
            [
                [
                    [1.0, 2.5, -0.1],
                    [0.1, 0.0, 3.0],
                    [0.002, -6.0, -0.5],
                ],
            ],
        ];
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals('Noise (std dev: 0.1)', (string) $this->layer);
    }

    #[Test]
    #[TestDox('Constructor rejects invalid standard deviation')]
    public function testConstructorRejectsInvalidStdDev() : void
    {
        $this->expectException(InvalidArgumentException::class);

        // Negative std dev should be rejected
        new Noise(-0.1);
    }

    #[Test]
    #[TestDox('Forward throws if layer is not initialized')]
    public function testForwardThrowsIfNotInitialized() : void
    {
        $layer = new Noise(0.1);

        $this->expectException(RuntimeException::class);

        $layer->forward($this->input);
    }

    #[Test]
    #[TestDox('Initializes width equal to fan-in')]
    public function testInitializeSetsWidth() : void
    {
        $this->layer->initialize($this->fanIn);

        self::assertEquals($this->fanIn, $this->layer->width());
    }

    #[Test]
    #[TestDox('Computes forward pass that adds Gaussian noise with correct shape and scale')]
    public function testForwardAddsNoiseWithCorrectProperties() : void
    {
        $this->layer->initialize($this->fanIn);

        $forward = $this->layer->forward($this->input);

        self::assertInstanceOf(NDArray::class, $forward);

        $inputArray = $this->input->toArray();
        $forwardArray = $forward->toArray();

        // 1) Shape is preserved
        self::assertSameSize($inputArray, $forwardArray);

        // 2) At least one element differs (very high probability)
        $allEqual = true;
        foreach ($inputArray as $i => $row) {
            if ($row !== $forwardArray[$i]) {
                $allEqual = false;
                break;
            }
        }
        self::assertFalse($allEqual, 'Expected forward output to differ from input due to noise.');

        // 3) Empirical std dev of (forward - input) is ~ stdDev, within tolerance
        $diffs = [];
        foreach ($inputArray as $i => $row) {
            foreach ($row as $j => $v) {
                $diffs[] = $forwardArray[$i][$j] - $v;
            }
        }

        $n = count($diffs);
        $mean = array_sum($diffs) / $n;

        $var = 0.0;
        foreach ($diffs as $d) {
            $var += ($d - $mean) * ($d - $mean);
        }
        $var /= $n;
        $std = sqrt($var);

        // Mean of noise should be near 0, std near $this->stdDev
        self::assertEqualsWithDelta(0.0, $mean, 2e-1);   // +/-0.2 around 0
        self::assertEqualsWithDelta(0.1, $std, 1e-1);    // +/-0.1 around 0.1
    }

    #[Test]
    #[TestDox('Backpropagates and returns previous gradient unchanged')]
    #[DataProvider('backProvider')]
    public function testBackReturnsPrevGradient(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);
        $this->layer->forward($this->input);

        $gradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        self::assertInstanceOf(NDArray::class, $gradient);
        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Infer returns input unchanged')]
    #[DataProvider('inferProvider')]
    public function testInferIdentity(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $infer = $this->layer->infer($this->input);

        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }
}
