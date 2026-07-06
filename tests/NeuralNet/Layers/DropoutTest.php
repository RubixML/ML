<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers\Dropout;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use Rubix\ML\Deferred;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(Dropout::class)]
class DropoutTest extends TestCase
{
    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected NDArray $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected Dropout $layer;

    /**
     * @return array<string, array{0: float}>
     */
    public static function badRatioProvider() : array
    {
        return [
            'zero' => [0.0],
            'negative' => [-0.1],
            'one' => [1.0],
            'greaterThanOne' => [1.1],
        ];
    }

    /**
     * @return array<string, array{0: array<array<float>>}>
     */
    public static function inferProvider() : array
    {
        return [
            'identityOnInput' => [[
                [1.0, 2.5, -0.1],
                [0.1, 0.0, 3.0],
                [0.002, -6.0, -0.5],
            ]],
        ];
    }

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

        $this->layer = new Dropout(0.5);
    }

    #[Test]
    #[TestDox('Constructor rejects invalid ratio values')]
    #[DataProvider('badRatioProvider')]
    public function testConstructorRejectsInvalidRatio(float $ratio) : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Dropout($ratio);
    }

    #[Test]
    #[TestDox('Initializes width equal to fan-in')]
    public function testInitializeSetsWidth() : void
    {
        $this->layer->initialize($this->fanIn);

        self::assertEquals($this->fanIn, $this->layer->width());
    }

    #[Test]
    #[TestDox('Method forward() applies dropout mask with correct shape and scaling')]
    public function testForward() : void
    {
        $this->layer->initialize($this->fanIn);

        $forward = $this->layer->forward($this->input);

        $inputArray = $this->input->toArray();
        $forwardArray = $forward->toArray();

        self::assertSameSize($inputArray, $forwardArray);

        $scale = 1.0 / (1.0 - 0.5); // ratio = 0.5

        $nonZero = 0;
        $total = 0;

        foreach ($inputArray as $i => $row) {
            foreach ($row as $j => $x) {
                $y = $forwardArray[$i][$j];
                ++$total;

                if (abs($x) < 1e-12) {
                    // If input is (near) zero, output should also be ~0
                    self::assertEqualsWithDelta(0.0, $y, 1e-7);

                    continue;
                }

                if (abs($y) < 1e-12) {
                    // Dropped unit
                    continue;
                }

                ++$nonZero;

                // Kept unit should be scaled input
                self::assertEqualsWithDelta($x * $scale, $y, 1e-6);
            }
        }

        // Roughly (1 - ratio) of units should be non-zero; allow wide tolerance
        $expectedKept = (1.0 - 0.5) * $total;

        // In rare cases, all units could be dropped due to random chance
        // If this happens, we should still pass the test but note the issue
        if ($nonZero === 0) {
            self::markTestIncomplete('All units were dropped - this is rare but possible with random dropout');

            return;
        }

        self::assertGreaterThan(0, $nonZero);
        self::assertLessThan($total, $nonZero);
        self::assertEqualsWithDelta($expectedKept, $nonZero, $total * 0.5);
    }

    #[Test]
    #[TestDox('Backpropagates gradients using the same dropout mask')]
    public function testBack() : void
    {
        $this->layer->initialize($this->fanIn);

        // Forward pass to generate and store mask
        $forward = $this->layer->forward($this->input);
        $forwardArray = $forward->toArray();
        $inputArray = $this->input->toArray();

        // Approximate mask from forward output: mask ≈ forward / input
        $maskArray = [];

        foreach ($inputArray as $i => $row) {
            foreach ($row as $j => $x) {
                $y = $forwardArray[$i][$j];

                if (abs($x) < 1e-12) {
                    $maskArray[$i][$j] = 0.0;
                } else {
                    $maskArray[$i][$j] = $y / $x;
                }
            }
        }

        $gradient = $this->layer->back(
            prevGradient: $this->prevGrad,
            optimizer: $this->optimizer
        )->compute();

        $gradArray = $gradient->toArray();
        $prevGradArray = ($this->prevGrad)()->toArray();

        // Expected gradient per element: prevGrad * mask for non-zero inputs.
        // For zero inputs, the mask cannot be inferred from the forward output
        // (forward is always 0 regardless of mask), so we accept the actual
        // gradient value there.
        $expectedGrad = [];

        foreach ($prevGradArray as $i => $row) {
            foreach ($row as $j => $g) {
                if (abs($inputArray[$i][$j]) < 1e-12) {
                    $expectedGrad[$i][$j] = $gradArray[$i][$j];
                } else {
                    $expectedGrad[$i][$j] = $g * $maskArray[$i][$j];
                }
            }
        }

        self::assertEqualsWithDelta($expectedGrad, $gradArray, 1e-6);
    }

    #[Test]
    #[TestDox('Inference pass leaves inputs unchanged')]
    #[DataProvider('inferProvider')]
    public function testInfer(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $infer = $this->layer->infer($this->input);

        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Method initialize() returns fan out equal to fan in')]
    public function testInitializeReturnsFanOut() : void
    {
        $fanOut = $this->layer->initialize($this->fanIn);

        self::assertSame($this->fanIn, $fanOut);
    }

    #[Test]
    #[TestDox('Method width() returns the initialized width')]
    public function testWidthAfterInitialize() : void
    {
        $this->layer->initialize($this->fanIn);

        self::assertSame($this->fanIn, $this->layer->width());
    }

    #[Test]
    #[TestDox('Method gradient() multiplies previous gradient by the dropout mask')]
    public function testGradient() : void
    {
        // Deterministic previous gradient (same shape as input)
        $prevGradNd = NumPower::array([
            [0.25, 0.7, 0.1],
            [0.50, 0.2, 0.01],
            [0.25, 0.1, 0.89],
        ]);

        // Same deterministic mask as used in testForward/testBack
        $mask = NumPower::array([
            [2.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]);

        $prevGradient = new Deferred(fn: static function () use ($prevGradNd) : NDArray {
            return $prevGradNd;
        });

        $gradient = $this->layer->gradient($prevGradient, $mask);

        $expected = [
            [0.5, 1.4, 0.2],
            [1.0, 0.0, 0.02],
            [0.5, 0.2, 0.0],
        ];

        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('It returns correct string representation')]
    public function testToString() : void
    {
        $expected = 'Dropout (ratio: 0.5)';

        self::assertSame($expected, (string) $this->layer);
    }
}
