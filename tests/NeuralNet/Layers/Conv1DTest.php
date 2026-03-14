<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Conv1D;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use PHPUnit\Framework\TestCase;

/**
 * @group Layers
 * @covers \Rubix\ML\NeuralNet\Layers\Conv1D
 */
class Conv1DTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected int $inputLength = 10;

    /**
     * @var positive-int
     */
    protected int $inputChannels = 1;

    /**
     * @var Matrix
     */
    protected Matrix $input;

    /**
     * @var Deferred
     */
    protected Deferred $prevGrad;

    /**
     * @var Stochastic
     */
    protected Stochastic $optimizer;

    /**
     * @var Conv1D
     */
    protected Conv1D $layer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        // Create input: shape (inputChannels, inputLength * batchSize)
        // Here we use batchSize = 3
        $batchSize = 3;
        $inputData = [];

        for ($c = 0; $c < $this->inputChannels; ++$c) {
            $row = [];

            for ($b = 0; $b < $batchSize; ++$b) {
                for ($t = 0; $t < $this->inputLength; ++$t) {
                    $row[] = $t + $b * $this->inputLength + $c * 0.1;
                }
            }
            $inputData[] = $row;
        }

        $this->input = Matrix::quick($inputData);

        // Create gradient for backprop: shape (filters, outputLength * batchSize)
        $filters = 2;
        $kernelSize = 3;
        $outputLength = $this->inputLength - $kernelSize + 1; // 8

        $gradData = [];

        for ($f = 0; $f < $filters; ++$f) {
            $row = [];

            for ($b = 0; $b < $batchSize; ++$b) {
                for ($t = 0; $t < $outputLength; ++$t) {
                    $row[] = 0.1 * ($t + $f);
                }
            }
            $gradData[] = $row;
        }

        $this->prevGrad = new Deferred(function () use ($gradData) {
            return Matrix::quick($gradData);
        });

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new Conv1D(
            $filters,
            $kernelSize,
            $this->inputLength,
            $this->inputChannels,
            1, // stride
            0, // padding
            0.0, // l2 penalty
            true, // bias
            new He(),
            new Constant(0.0)
        );

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Conv1D::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    /**
     * @test
     */
    public function initializeForwardBackInfer() : void
    {
        $fanOut = $this->layer->initialize($this->inputLength);

        // Width should be number of filters
        $this->assertEquals(2, $this->layer->width());

        // Output length should be inputLength - kernelSize + 1 = 10 - 3 + 1 = 8
        $this->assertEquals(8, $this->layer->outputLength());

        // Fan out should be filters * outputLength = 2 * 8 = 16
        $this->assertEquals(16, $fanOut);

        // Forward pass
        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);

        // Output shape should be (filters, outputLength * batchSize) = (2, 8 * 3) = (2, 24)
        $this->assertEquals(2, $forward->m());
        $this->assertEquals(24, $forward->n());

        // Backward pass
        $gradient = $this->layer->back($this->prevGrad, $this->optimizer)->compute();

        $this->assertInstanceOf(Matrix::class, $gradient);

        // Gradient shape should match input shape
        $this->assertEquals($this->inputChannels, $gradient->m());
        $this->assertEquals($this->input->n(), $gradient->n());

        // Inference pass - should produce same shape as forward
        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals(2, $infer->m());
        $this->assertEquals(24, $infer->n());
    }

    /**
     * @test
     */
    public function withPadding() : void
    {
        // With padding=1, output length should be same as input length (for kernelSize=3)
        // outputLength = floor((inputLength + 2*padding - kernelSize) / stride) + 1
        //              = floor((10 + 2*1 - 3) / 1) + 1 = 10

        $layer = new Conv1D(
            2, // filters
            3, // kernelSize
            10, // inputLength
            1, // inputChannels
            1, // stride
            1, // padding
            0.0,
            true,
            new He(),
            new Constant(0.0)
        );

        srand(self::RANDOM_SEED);

        $layer->initialize(10);

        $this->assertEquals(10, $layer->outputLength());

        $forward = $layer->forward($this->input);

        // Output shape: (2, 10 * 3) = (2, 30)
        $this->assertEquals(2, $forward->m());
        $this->assertEquals(30, $forward->n());
    }

    /**
     * @test
     */
    public function withStride() : void
    {
        // With stride=2, output length should be half (roughly)
        // outputLength = floor((inputLength - kernelSize) / stride) + 1
        //              = floor((10 - 3) / 2) + 1 = 4

        $layer = new Conv1D(
            2, // filters
            3, // kernelSize
            10, // inputLength
            1, // inputChannels
            2, // stride
            0, // padding
            0.0,
            true,
            new He(),
            new Constant(0.0)
        );

        srand(self::RANDOM_SEED);

        $layer->initialize(10);

        $this->assertEquals(4, $layer->outputLength());

        $forward = $layer->forward($this->input);

        // Output shape: (2, 4 * 3) = (2, 12)
        $this->assertEquals(2, $forward->m());
        $this->assertEquals(12, $forward->n());
    }

    /**
     * @test
     */
    public function multiChannel() : void
    {
        // Test with 3 input channels
        $inputChannels = 3;
        $batchSize = 2;

        $inputData = [];

        for ($c = 0; $c < $inputChannels; ++$c) {
            $row = [];

            for ($b = 0; $b < $batchSize; ++$b) {
                for ($t = 0; $t < $this->inputLength; ++$t) {
                    $row[] = $t + $b * $this->inputLength + $c * 0.1;
                }
            }
            $inputData[] = $row;
        }

        $input = Matrix::quick($inputData);

        $layer = new Conv1D(
            4, // filters
            3, // kernelSize
            10, // inputLength
            $inputChannels, // inputChannels
            1, // stride
            0, // padding
            0.0,
            true,
            new He(),
            new Constant(0.0)
        );

        srand(self::RANDOM_SEED);

        $layer->initialize(10);

        $forward = $layer->forward($input);

        // Output shape: (4, 8 * 2) = (4, 16)
        $this->assertEquals(4, $forward->m());
        $this->assertEquals(16, $forward->n());

        // Backward pass
        $gradData = [];

        for ($f = 0; $f < 4; ++$f) {
            $row = [];

            for ($b = 0; $b < $batchSize; ++$b) {
                for ($t = 0; $t < 8; ++$t) {
                    $row[] = 0.1 * ($t + $f);
                }
            }
            $gradData[] = $row;
        }

        $prevGrad = new Deferred(function () use ($gradData) {
            return Matrix::quick($gradData);
        });

        $optimizer = new Stochastic(0.001);

        $gradient = $layer->back($prevGrad, $optimizer)->compute();

        // Gradient shape should match input: (3, 10 * 2) = (3, 20)
        $this->assertEquals($inputChannels, $gradient->m());
        $this->assertEquals($input->n(), $gradient->n());
    }

    /**
     * @test
     */
    public function invalidParameters() : void
    {
        // Test kernel size larger than input length with no padding
        $this->expectException(\Rubix\ML\Exceptions\InvalidArgumentException::class);

        new Conv1D(
            2, // filters
            15, // kernelSize > inputLength
            10, // inputLength
            1, // inputChannels
            1, // stride
            0 // padding
        );
    }

    /**
     * @test
     */
    public function negativeStride() : void
    {
        $this->expectException(\Rubix\ML\Exceptions\InvalidArgumentException::class);

        new Conv1D(
            2, // filters
            3, // kernelSize
            10, // inputLength
            1, // inputChannels
            0, // stride (invalid)
            0 // padding
        );
    }

    /**
     * @test
     */
    public function negativePadding() : void
    {
        $this->expectException(\Rubix\ML\Exceptions\InvalidArgumentException::class);

        new Conv1D(
            2, // filters
            3, // kernelSize
            10, // inputLength
            1, // inputChannels
            1, // stride
            -1 // padding (invalid)
        );
    }

    /**
     * @test
     */
    public function l2Penalty() : void
    {
        $layer = new Conv1D(
            2, // filters
            3, // kernelSize
            10, // inputLength
            1, // inputChannels
            1, // stride
            0, // padding
            0.01, // l2 penalty
            true,
            new He(),
            new Constant(0.0)
        );

        srand(self::RANDOM_SEED);

        $layer->initialize(10);

        $forward = $layer->forward($this->input);

        $gradData = [];

        for ($f = 0; $f < 2; ++$f) {
            $row = [];

            for ($b = 0; $b < 3; ++$b) {
                for ($t = 0; $t < 8; ++$t) {
                    $row[] = 0.1;
                }
            }
            $gradData[] = $row;
        }

        $prevGrad = new Deferred(function () use ($gradData) {
            return Matrix::quick($gradData);
        });

        $optimizer = new Stochastic(0.001);

        $gradient = $layer->back($prevGrad, $optimizer)->compute();

        $this->assertInstanceOf(Matrix::class, $gradient);
    }

    /**
     * @test
     */
    public function noBias() : void
    {
        $layer = new Conv1D(
            2, // filters
            3, // kernelSize
            10, // inputLength
            1, // inputChannels
            1, // stride
            0, // padding
            0.0,
            false, // no bias
            new He(),
            new Constant(0.0)
        );

        srand(self::RANDOM_SEED);

        $layer->initialize(10);

        $forward = $layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);

        // Check that there's no bias parameter
        $params = iterator_to_array($layer->parameters());
        $this->assertArrayHasKey('weights', $params);
        $this->assertArrayNotHasKey('biases', $params);
    }

    /**
     * @test
     */
    public function forwardPassCalculation() : void
    {
        $layer = new Conv1D(
            1,
            3,
            5,
            1,
            1,
            0,
            0.0,
            false,
            new Constant(0.1),
            new Constant(0.0)
        );

        $layer->initialize(5);

        $weights = $layer->weights();
        $this->assertEqualsWithDelta(0.1, $weights[0][0], 1e-10);
        $this->assertEqualsWithDelta(0.1, $weights[0][1], 1e-10);
        $this->assertEqualsWithDelta(0.1, $weights[0][2], 1e-10);

        $input = Matrix::quick([[1.0, 2.0, 3.0, 4.0, 5.0]]);

        $output = $layer->forward($input);
        $outputArray = $output->asArray()[0];

        $this->assertEqualsWithDelta(0.6, $outputArray[0], 1e-10);
        $this->assertEqualsWithDelta(0.9, $outputArray[1], 1e-10);
        $this->assertEqualsWithDelta(1.2, $outputArray[2], 1e-10);
    }

    /**
     * @test
     */
    public function backwardPassCalculation() : void
    {
        $layer = new Conv1D(
            1,
            3,
            5,
            1,
            1,
            0,
            0.0,
            false,
            new Constant(0.1),
            new Constant(0.0)
        );

        $layer->initialize(5);
        $weights = $layer->weights();
        $this->assertEqualsWithDelta(0.1, $weights[0][0], 1e-10);
        $this->assertEqualsWithDelta(0.1, $weights[0][1], 1e-10);
        $this->assertEqualsWithDelta(0.1, $weights[0][2], 1e-10);

        $input = Matrix::quick([[1.0, 2.0, 3.0, 4.0, 5.0]]);

        $layer->forward($input);
        $dOut = Matrix::quick([[0.5, 0.3, 0.2]]);

        $prevGrad = new Deferred(function () use ($dOut) {
            return $dOut;
        });

        $optimizer = new Stochastic(0.001);
        $dInput = $layer->back($prevGrad, $optimizer)->compute();
        $dInputArray = $dInput->asArray()[0];

        $this->assertEquals(5, count($dInputArray));
        $this->assertEqualsWithDelta(0.05, $dInputArray[0], 1e-10);
        $this->assertEqualsWithDelta(0.08, $dInputArray[1], 1e-10);
        $this->assertEqualsWithDelta(0.10, $dInputArray[2], 1e-10);
        $this->assertEqualsWithDelta(0.05, $dInputArray[3], 1e-10);
        $this->assertEqualsWithDelta(0.02, $dInputArray[4], 1e-10);
    }

    /**
     * @test
     */
    public function gradientNumericalCheck() : void
    {
        $kernelSize = 3;
        $inputLength = 5;
        $filters = 1;
        $inputChannels = 1;

        $layer = new Conv1D(
            $filters,
            $kernelSize,
            $inputLength,
            $inputChannels,
            1,
            0,
            0.0,
            false,
            new Constant(0.0),
            new Constant(0.0)
        );

        $layer->initialize($inputLength);
        $inputData = [[1.0, 2.0, 3.0, 4.0, 5.0]];
        $input = Matrix::quick($inputData);

        $layer->forward($input);

        $dOutData = [[1.0, 1.0, 1.0]];
        $dOut = Matrix::quick($dOutData);

        $prevGrad = new Deferred(function () use ($dOut) {
            return $dOut;
        });

        $optimizer = new Stochastic(0.001);

        $dInput = $layer->back($prevGrad, $optimizer)->compute();
        $analyticalGrad = $dInput->asArray()[0];

        $epsilon = 1e-5;
        $numericalGrad = [];

        for ($i = 0; $i < $inputLength; ++$i) {
            $plus = $inputData;
            $minus = $inputData;
            $plus[0][$i] += $epsilon;
            $minus[0][$i] -= $epsilon;

            $layerPlus = new Conv1D($filters, $kernelSize, $inputLength, $inputChannels, 1, 0, 0.0, false, new Constant(0.0), new Constant(0.0));
            $layerPlus->initialize($inputLength);
            $outPlus = $layerPlus->forward(Matrix::quick($plus));

            $layerMinus = new Conv1D($filters, $kernelSize, $inputLength, $inputChannels, 1, 0, 0.0, false, new Constant(0.0), new Constant(0.0));
            $layerMinus->initialize($inputLength);
            $outMinus = $layerMinus->forward(Matrix::quick($minus));

            $sumPlus = array_sum($outPlus->asArray()[0]);
            $sumMinus = array_sum($outMinus->asArray()[0]);

            $numericalGrad[$i] = ($sumPlus - $sumMinus) / (2 * $epsilon);
        }

        for ($i = 0; $i < $inputLength; ++$i) {
            $this->assertEqualsWithDelta(
                $numericalGrad[$i],
                $analyticalGrad[$i],
                1e-4,
                "Input gradient mismatch at index {$i}: numerical={$numericalGrad[$i]}, analytical={$analyticalGrad[$i]}"
            );
        }
    }
}
