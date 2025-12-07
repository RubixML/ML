<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Layers\BatchNorm;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\Attributes\DataProvider;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\BatchNorm\BatchNorm;
use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant\Constant;
use Rubix\ML\NeuralNet\Parameters\Parameter as TrainableParameter;
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
                [-0.1251222, 1.2825030, -1.1573808],
                [-0.6708631, -0.7427414, 1.4136046],
                [0.7974157, -1.4101899, 0.6127743],
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

        for ($i = 0; $i < $rows; $i++) {
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
}
