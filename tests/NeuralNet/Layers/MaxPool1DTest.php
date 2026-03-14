<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\MaxPool1D;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use PHPUnit\Framework\TestCase;

/**
 * @group Layers
 * @covers \Rubix\ML\NeuralNet\Layers\MaxPool1D
 */
class MaxPool1DTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected int $inputLength = 10;

    /**
     * @var positive-int
     */
    protected int $inputChannels = 2;

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
     * @var MaxPool1D
     */
    protected MaxPool1D $layer;

    protected int $batchSize = 3;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $inputData = [];

        for ($c = 0; $c < $this->inputChannels; ++$c) {
            $row = [];

            for ($b = 0; $b < $this->batchSize; ++$b) {
                for ($t = 0; $t < $this->inputLength; ++$t) {
                    $row[] = $t + $b * $this->inputLength + $c * 0.1;
                }
            }

            $inputData[] = $row;
        }

        $this->input = Matrix::quick($inputData);

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new MaxPool1D(
            3,
            $this->inputLength,
            1
        );

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(MaxPool1D::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
    }

    /**
     * @test
     */
    public function initializeForwardBackInfer() : void
    {
        $fanOut = $this->layer->initialize($this->inputChannels);

        // Width should be number of input channels
        $this->assertEquals($this->inputChannels, $this->layer->width());

        // Output length: floor((10 - 3) / 1) + 1 = 8
        $this->assertEquals(8, $this->layer->outputLength());

        // Fan out should be inputChannels * outputLength = 2 * 8 = 16
        $this->assertEquals(16, $fanOut);

        // Forward pass
        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);

        // Output shape: (inputChannels, outputLength * batchSize) = (2, 8 * 3) = (2, 24)
        $this->assertEquals($this->inputChannels, $forward->m());
        $this->assertEquals(8 * $this->batchSize, $forward->n());

        // Backward pass
        $outputLength = 8;
        $gradData = [];

        for ($c = 0; $c < $this->inputChannels; ++$c) {
            $row = [];

            for ($b = 0; $b < $this->batchSize; ++$b) {
                for ($t = 0; $t < $outputLength; ++$t) {
                    $row[] = 0.01 * ($t + $c);
                }
            }

            $gradData[] = $row;
        }

        $this->prevGrad = new Deferred(function () use ($gradData) {
            return Matrix::quick($gradData);
        });

        $gradient = $this->layer->back($this->prevGrad, $this->optimizer)->compute();

        $this->assertInstanceOf(Matrix::class, $gradient);

        // Gradient shape should match input shape
        $this->assertEquals($this->inputChannels, $gradient->m());
        $this->assertEquals($this->inputLength * $this->batchSize, $gradient->n());

        // Inference pass
        $infer = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($this->inputChannels, $infer->m());
        $this->assertEquals(8 * $this->batchSize, $infer->n());
    }

    /**
     * @test
     */
    public function withStride() : void
    {
        // With stride=2, output length = floor((10 - 3) / 2) + 1 = 4
        $layer = new MaxPool1D(
            3,
            10,
            2
        );

        srand(self::RANDOM_SEED);

        $layer->initialize($this->inputChannels);

        $this->assertEquals(4, $layer->outputLength());

        $forward = $layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);

        // Output shape: (2, 4 * 3) = (2, 12)
        $this->assertEquals($this->inputChannels, $forward->m());
        $this->assertEquals(4 * $this->batchSize, $forward->n());
    }

    /**
     * @test
     */
    public function defaultStride() : void
    {
        // With default stride (= poolSize), output length = floor((10 - 3) / 3) + 1 = 3
        $layer = new MaxPool1D(
            3,
            10
        );

        srand(self::RANDOM_SEED);

        $layer->initialize($this->inputChannels);

        $this->assertEquals(3, $layer->outputLength());

        $forward = $layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($this->inputChannels, $forward->m());
        $this->assertEquals(3 * $this->batchSize, $forward->n());
    }

    /**
     * @test
     */
    public function invalidPoolSize() : void
    {
        $this->expectException(\Rubix\ML\Exceptions\InvalidArgumentException::class);

        new MaxPool1D(
            15,
            10
        );
    }

    /**
     * @test
     */
    public function negativeStride() : void
    {
        $this->expectException(\Rubix\ML\Exceptions\InvalidArgumentException::class);

        new MaxPool1D(
            3,
            10,
            -1
        );
    }

    /**
     * @test
     */
    public function gradientNumericalCheck() : void
    {
        $layer = new MaxPool1D(2, 4, 1);
        $layer->initialize(1);

        $inputData = [[1.0, 3.0, 2.0, 4.0]];
        $input = Matrix::quick($inputData);
        $output = $layer->forward($input);

        $outputArray = $output->asArray()[0];
        $this->assertEquals(3.0, $outputArray[0]); // max(1, 3) = 3
        $this->assertEquals(3.0, $outputArray[1]); // max(3, 2) = 3
        $this->assertEquals(4.0, $outputArray[2]); // max(2, 4) = 4

        $gradData = [[1.0, 1.0, 1.0]];
        $prevGrad = new Deferred(function () use ($gradData) {
            return Matrix::quick($gradData);
        });

        $gradient = $layer->back($prevGrad, $this->optimizer)->compute();
        $analyticalGrad = $gradient->asArray()[0];

        // Expected gradients: [0, 2, 0, 1] (sum of gradients at max positions)
        $this->assertEquals(0.0, $analyticalGrad[0]);
        $this->assertEquals(2.0, $analyticalGrad[1]);
        $this->assertEquals(0.0, $analyticalGrad[2]);
        $this->assertEquals(1.0, $analyticalGrad[3]);

        $epsilon = 1e-5;
        $numericalGrad = [];

        for ($i = 0; $i < 4; ++$i) {
            $plus = $inputData;
            $minus = $inputData;
            $plus[0][$i] += $epsilon;
            $minus[0][$i] -= $epsilon;

            $layerPlus = new MaxPool1D(2, 4, 1);
            $layerPlus->initialize(1);
            $outPlus = $layerPlus->forward(Matrix::quick($plus));

            $layerMinus = new MaxPool1D(2, 4, 1);
            $layerMinus->initialize(1);
            $outMinus = $layerMinus->forward(Matrix::quick($minus));

            $sumPlus = array_sum($outPlus->asArray()[0]);
            $sumMinus = array_sum($outMinus->asArray()[0]);

            $numericalGrad[$i] = ($sumPlus - $sumMinus) / (2 * $epsilon);
        }

        for ($i = 0; $i < 4; ++$i) {
            $this->assertEqualsWithDelta(
                $numericalGrad[$i],
                $analyticalGrad[$i],
                1e-4,
                "Gradient mismatch at index {$i}: numerical={$numericalGrad[$i]}, analytical={$analyticalGrad[$i]}"
            );
        }
    }
}
