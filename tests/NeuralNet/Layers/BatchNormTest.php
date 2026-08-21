<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\Attributes\DataProvider;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Parameter as TrainableParameter;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException as RubixRuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Layers')]
#[CoversClass(BatchNorm::class)]
class BatchNormTest extends TestCase
{
    /**
     * @var positive-int
     */
    protected int $fanIn;

    protected NDArray $input;

    protected Deferred $prevGrad;

    protected Optimizer $optimizer;

    protected BatchNorm $layer;

    /**
     * @return array<string, array{0:int}>
     */
    public static function initializeProvider() : array
    {
        return [
            'fanIn=3' => [3],
        ];
    }

    /**
     * @return array<string, array{0:array}>
     */
    public static function forwardProvider() : array
    {
        return [
            'expectedForward' => [[
                [-0.1251222, 1.2825031, -1.1573808],
                [-0.6708631, -0.7427414, 1.4136046],
                [0.7974158, -1.4101899, 0.6127743],
            ]],
        ];
    }

    /**
     * @return array<string, array{0:array}>
     */
    public static function backProvider() : array
    {
        return [
            'expectedGradient' => [[
                [-0.0644587, 0.0272710, 0.0371877],
                [0.1137590, -0.1099670, -0.0037919],
                [-0.1190978, -0.0108703, 0.1299681],
            ]],
        ];
    }

    /**
     * @return array<string, array{0:array}>
     */
    public static function inferProvider() : array
    {
        return [
            'expectedInfer' => [[
                [-0.1251222, 1.2825031, -1.1573808],
                [-0.6708631, -0.7427414, 1.4136046],
                [0.7974158, -1.4101899, 0.6127743],
            ]],
        ];
    }

    /**
     * Additional inputs to validate behavior across different batch sizes.
     *
     * @return array<string, array{0:array}>
     */
    public static function batchInputsProvider() : array
    {
        return [
            'batch1x3' => [[
                [2.0, -1.0, 0.0],
            ]],
            'batch2x3' => [[
                [1.0, 2.0, 3.0],
                [3.0, 3.0, 3.0],
            ]],
            'batch4x3' => [[
                [0.5, -0.5, 1.5],
                [10.0, -10.0, 0.0],
                [7.2, 3.3, -2.4],
                [-1.0, -2.0, 4.0],
            ]],
        ];
    }

    /**
     * @return array<string, array{0:array}>
     */
    public static function gradientProvider() : array
    {
        return [
            'expectedGradient' => [[
                [-0.0644587, 0.0272710, 0.0371877],
                [0.1137590, -0.1099670, -0.0037919],
                [-0.1190978, -0.0108703, 0.1299681],
            ]],
        ];
    }

    /**
     * @return array<string, array{0:float}>
     */
    public static function badDecayProvider() : array
    {
        return [
            'negative' => [-0.01],
            'greaterThanOne' => [1.01],
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

        $this->layer = new BatchNorm(
            decay: 0.9,
            betaInitializer: new Constant(0.0),
            gammaInitializer: new Constant(1.0)
        );
    }

    #[Test]
    #[TestDox('Can be cast to a string')]
    public function testToString() : void
    {
        self::assertEquals(
            'Batch Norm (decay: 0.9, beta initializer: Constant (value: 0), gamma initializer: Constant (value: 1))',
            (string) $this->layer
        );
    }

    #[Test]
    #[TestDox('Initializes width and returns fan out')]
    #[DataProvider('initializeProvider')]
    public function testInitialize(int $fanIn) : void
    {
        $fanOut = $this->layer->initialize($fanIn);
        self::assertEquals($fanIn, $fanOut);
        self::assertEquals($fanIn, $this->layer->width());
    }

    #[Test]
    #[TestDox('Computes forward pass')]
    #[DataProvider('forwardProvider')]
    public function testForward(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        $forward = $this->layer->forward($this->input);

        self::assertEqualsWithDelta($expected, $forward->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Backpropagates and returns gradient for previous layer')]
    #[DataProvider('backProvider')]
    public function testBack(array $expected) : void
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
    #[TestDox('Infers using running statistics')]
    #[DataProvider('inferProvider')]
    public function testInfer(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);
        // Perform a forward pass to set running mean/variance
        $this->layer->forward($this->input);

        $infer = $this->layer->infer($this->input);

        self::assertEqualsWithDelta($expected, $infer->toArray(), 1e-7);
    }

    #[Test]
    #[TestDox('Computes forward pass (row-wise) with zero mean and unit variance per sample for various batch sizes')]
    #[DataProvider('batchInputsProvider')]
    public function testForwardStatsMultipleBatches(array $input) : void
    {
        $this->layer->initialize($this->fanIn);

        $forward = $this->layer->forward(NumPower::array($input));
        $out = $forward->toArray();

        // Check per-row mean ~ 0 and variance ~ 1 (allow 0 for degenerate rows)
        $this->assertRowwiseStats($input, $out, true);
    }

    #[Test]
    #[TestDox('Infers (row-wise) with zero mean and unit variance per sample for various batch sizes')]
    #[DataProvider('batchInputsProvider')]
    public function testInferStatsMultipleBatches(array $input) : void
    {
        $this->layer->initialize($this->fanIn);

        // Perform a forward pass on the same input to initialize running stats
        $this->layer->forward(NumPower::array($input));

        $infer = $this->layer->infer(NumPower::array($input));
        $out = $infer->toArray();

        $this->assertRowwiseStats($input, $out, false);
    }

    #[Test]
    #[TestDox('Throws when width is requested before initialization')]
    public function testWidthThrowsBeforeInitialize() : void
    {
        $layer = new BatchNorm();
        $this->expectException(RubixRuntimeException::class);
        $layer->width();
    }

    #[Test]
    #[TestDox('Constructor rejects invalid decay values')]
    #[DataProvider('badDecayProvider')]
    public function testConstructorRejectsInvalidDecay(float $decay) : void
    {
        $this->expectException(InvalidArgumentException::class);
        new BatchNorm(decay: $decay);
    }

    #[Test]
    #[TestDox('Yields trainable parameters beta and gamma')]
    public function testParameters() : void
    {
        $this->layer->initialize($this->fanIn);

        $params = iterator_to_array($this->layer->parameters());

        self::assertArrayHasKey('beta', $params);
        self::assertArrayHasKey('gamma', $params);
        self::assertInstanceOf(TrainableParameter::class, $params['beta']);
        self::assertInstanceOf(TrainableParameter::class, $params['gamma']);

        self::assertEquals([0.0, 0.0, 0.0], $params['beta']->param()->toArray());
        self::assertEquals([1.0, 1.0, 1.0], $params['gamma']->param()->toArray());
    }

    #[Test]
    #[TestDox('Restores parameters from array')]
    public function testRestore() : void
    {
        $this->layer->initialize($this->fanIn);

        $betaNew = new TrainableParameter(NumPower::full([3], 2.0));
        $gammaNew = new TrainableParameter(NumPower::full([3], 3.0));

        $this->layer->restore([
            'beta' => $betaNew,
            'gamma' => $gammaNew,
        ]);

        $restored = iterator_to_array($this->layer->parameters());
        self::assertSame($betaNew, $restored['beta']);
        self::assertSame($gammaNew, $restored['gamma']);
        self::assertEquals([2.0, 2.0, 2.0], $restored['beta']->param()->toArray());
        self::assertEquals([3.0, 3.0, 3.0], $restored['gamma']->param()->toArray());
    }

    #[Test]
    #[TestDox('Computes gradient for previous layer directly')]
    #[DataProvider('gradientProvider')]
    public function testGradient(array $expected) : void
    {
        $this->layer->initialize($this->fanIn);

        // Compute forward-time caches manually to pass into gradient()
        $input = $this->input;
        $rows = $input->shape()[0];
        $meanArr = [];
        $varArr = [];
        $stdInvArr = [];

        for ($i = 0; $i < $rows; ++$i) {
            $row = $input->toArray()[$i];
            $meanArr[$i] = NumPower::mean($row);
            $varArr[$i] = NumPower::variance($row);
            $stdInvArr[$i] = 1.0 / sqrt($varArr[$i]);
        }

        $mean = NumPower::array($meanArr);
        $stdInv = NumPower::array($stdInvArr);

        $xHat = NumPower::multiply(
            NumPower::subtract(NumPower::transpose($input, [1, 0]), $mean),
            $stdInv
        );
        $xHat = NumPower::transpose($xHat, [1, 0]);

        // Use provided prevGrad as dOut and current gamma parameter
        $dOut = ($this->prevGrad)();
        $gamma = iterator_to_array($this->layer->parameters())['gamma']->param();

        $gradient = $this->layer->gradient($dOut, $gamma, $stdInv, $xHat);

        self::assertEqualsWithDelta($expected, $gradient->toArray(), 1e-7);
    }

    /**
     * @test
     */
    public function normalizesOverBatchSize() : void
    {
        $fanIn = 3;

        $input = Matrix::quick([
            [1.0, 2.5, -0.1, 0.5],
            [0.1, 0.0, 3.0, -1.0],
            [0.002, -6.0, -0.5, 2.0],
        ]);

        $prevGrad = new Deferred(function () {
            return Matrix::quick([
                [0.25, 0.7, 0.1, 0.3],
                [0.50, 0.2, 0.01, -0.4],
                [0.25, 0.1, 0.89, 0.6],
            ]);
        });

        $optimizer = new Stochastic(0.001);

        $layer = new BatchNorm(0.9, new Constant(0.), new Constant(1.));

        $layer->initialize($fanIn);

        $expected = [
            [0.025967457200229, 1.584014889214, -1.1166006596098, -0.49338168680435],
            [-0.28480067232747, -0.35181259522806, 1.6585450917894, -1.0219318242339],
            [0.3797862163235, -1.643717441354, 0.21054282476168, 1.0533884002688],
        ];

        $forward = $layer->forward($input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        $expected = [
            [-0.096655673462753, 0.024584160424222, 0.0014008068617791, 0.070670706176752],
            [0.29326885034768, 0.094619781903042, -0.10430387932137, -0.28358475292935],
            [-0.094806324008858, -0.017466016738221, 0.13166046771019, -0.019388126963108],
        ];

        $gradient = $layer->back($prevGrad, $optimizer)->compute();

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        $expected = [
            [0.024595238724167, 1.5813095621742, -1.1169952651392, -0.49430953575917],
            [-0.28505012503587, -0.35204780151489, 1.6578824928559, -1.0220245663052],
            [0.37766138009295, -1.6443246681253, 0.20854491954554, 1.0507583684868],
        ];

        $infer = $layer->infer($input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }

    /**
     * @param array<int, array<int, float>> $inputRows
     * @param array<int, array<int, float>> $outRows
     * @param bool $checkMean
     */
    private function assertRowwiseStats(array $inputRows, array $outRows, bool $checkMean) : void
    {
        foreach ($outRows as $i => $row) {
            $mean = array_sum($row) / count($row);
            $var = 0.0;

            foreach ($row as $v) {
                $var += ($v - $mean) * ($v - $mean);
            }
            $var /= count($row);

            $orig = $inputRows[$i];
            $origMean = array_sum($orig) / count($orig);
            $origVar = 0.0;

            foreach ($orig as $ov) {
                $origVar += ($ov - $origMean) * ($ov - $origMean);
            }
            $origVar /= count($orig);

            $expectedVar = $origVar < 1e-12 ? 0.0 : 1.0;

            if ($checkMean) {
                self::assertEqualsWithDelta(0.0, $mean, 1e-7);
            }

            if ($expectedVar === 0.0) {
                self::assertLessThan(1e-6, $var);
            } else {
                self::assertEqualsWithDelta(1.0, $var, 1e-6);
            }
        }
    }
}
