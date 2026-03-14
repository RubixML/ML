<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * MaxPool1D
 *
 * A 1-dimensional max pooling layer that downsamples the input by taking the
 * maximum value over sliding windows. It is commonly used after convolutional
 * layers to reduce the sequence length while retaining the most prominent features.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Boorinio
 */
class MaxPool1D implements Hidden
{
    /**
     * The size of the pooling window.
     *
     * @var positive-int
     */
    protected int $poolSize;

    /**
     * The length of the input sequence.
     *
     * @var positive-int
     */
    protected int $inputLength;

    /**
     * The stride of the pooling operation.
     *
     * @var positive-int
     */
    protected int $stride;

    /**
     * The number of input channels.
     *
     * @var positive-int|null
     */
    protected ?int $inputChannels = null;

    /**
     * The computed output length.
     *
     * @var positive-int
     */
    protected int $outputLength;

    /**
     * The indices of the max values for backpropagation.
     *
     * @var array<int, array<int, int>>|null
     */
    protected ?array $maxIndices = null;

    /**
     * @param int $poolSize Size of the pooling window
     * @param int $inputLength Length of the input sequence
     * @param int $stride Stride of the pooling operation (default: same as poolSize)
     * @throws InvalidArgumentException
     */
    public function __construct(int $poolSize, int $inputLength, int $stride = 0)
    {
        if ($poolSize < 1) {
            throw new InvalidArgumentException('Pool size must be'
                . " greater than 0, $poolSize given.");
        }

        if ($inputLength < 1) {
            throw new InvalidArgumentException('Input length must be'
                . " greater than 0, $inputLength given.");
        }

        if ($stride < 0) {
            throw new InvalidArgumentException('Stride cannot be'
                . " negative, $stride given.");
        }

        $stride = $stride > 0 ? $stride : $poolSize;

        $outputLength = (int) floor(($inputLength - $poolSize) / $stride) + 1;

        if ($outputLength < 1) {
            throw new InvalidArgumentException('Output length must be'
                . " greater than 0, $outputLength given. Check pool size and stride values.");
        }

        $this->poolSize = $poolSize;
        $this->inputLength = $inputLength;
        $this->stride = $stride;
        $this->outputLength = $outputLength;
    }

    /**
     * Return the width of the layer (same as input channels).
     *
     * @internal
     *
     * @throws RuntimeException
     * @return positive-int
     */
    public function width() : int
    {
        if ($this->inputChannels === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        return $this->inputChannels;
    }

    /**
     * Return the output length after pooling.
     *
     * @internal
     *
     * @return positive-int
     */
    public function outputLength() : int
    {
        return $this->outputLength;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @internal
     *
     * @param positive-int $fanIn
     * @return positive-int
     */
    public function initialize(int $fanIn) : int
    {
        $this->inputChannels = $fanIn;

        return $fanIn * $this->outputLength;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @internal
     *
     * @param Matrix $input
     * @throws RuntimeException
     * @return Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($this->inputChannels === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $inputChannels = $input->m();

        if ($inputChannels !== $this->inputChannels) {
            throw new RuntimeException('Input channels mismatch:'
                . " expected {$this->inputChannels}, got {$inputChannels}.");
        }

        $batchSize = (int) ($input->n() / $this->inputLength);

        $inputArray = $input->asArray();
        $output = [];
        $maxIndices = [];

        foreach ($inputArray as $c => $channel) {
            $outputRow = [];
            $indicesRow = [];

            for ($b = 0; $b < $batchSize; ++$b) {
                $sampleOffset = $b * $this->inputLength;

                for ($t = 0; $t < $this->outputLength; ++$t) {
                    $startPos = $t * $this->stride;
                    $maxVal = -INF;
                    $maxIdx = $startPos;

                    for ($p = 0; $p < $this->poolSize; ++$p) {
                        $pos = $startPos + $p;
                        $val = $channel[$sampleOffset + $pos];

                        if ($val > $maxVal) {
                            $maxVal = $val;
                            $maxIdx = $sampleOffset + $pos;
                        }
                    }

                    $outputRow[] = $maxVal;
                    $indicesRow[] = $maxIdx;
                }
            }

            $output[] = $outputRow;
            $maxIndices[] = $indicesRow;
        }

        $this->maxIndices = $maxIndices;

        return Matrix::quick($output);
    }

    /**
     * Compute an inference pass through the layer.
     *
     * @internal
     *
     * @param Matrix $input
     * @throws RuntimeException
     * @return Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if ($this->inputChannels === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $inputChannels = $input->m();

        if ($inputChannels !== $this->inputChannels) {
            throw new RuntimeException('Input channels mismatch:'
                . " expected {$this->inputChannels}, got {$inputChannels}.");
        }

        $batchSize = (int) ($input->n() / $this->inputLength);

        $inputArray = $input->asArray();
        $output = [];

        foreach ($inputArray as $channel) {
            $outputRow = [];

            for ($b = 0; $b < $batchSize; ++$b) {
                $sampleOffset = $b * $this->inputLength;

                for ($t = 0; $t < $this->outputLength; ++$t) {
                    $startPos = $t * $this->stride;
                    $maxVal = -INF;

                    for ($p = 0; $p < $this->poolSize; ++$p) {
                        $pos = $startPos + $p;
                        $val = $channel[$sampleOffset + $pos];

                        if ($val > $maxVal) {
                            $maxVal = $val;
                        }
                    }

                    $outputRow[] = $maxVal;
                }
            }

            $output[] = $outputRow;
        }

        return Matrix::quick($output);
    }

    /**
     * Calculate the gradient and update the parameters of the layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if ($this->maxIndices === null) {
            throw new RuntimeException('Must perform forward pass before backpropagating.');
        }

        $maxIndices = $this->maxIndices;
        $inputChannels = $this->inputChannels;
        $inputLength = $this->inputLength;

        $this->maxIndices = null;

        return new Deferred([$this, 'gradient'], [$prevGradient, $maxIndices, $inputChannels, $inputLength]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param Deferred $prevGradient
     * @param array<int, array<int, int>> $maxIndices
     * @param int $inputChannels
     * @param int $inputLength
     * @return Matrix
     */
    public function gradient(
        Deferred $prevGradient,
        array $maxIndices,
        int $inputChannels,
        int $inputLength
    ) : Matrix {
        $dOut = $prevGradient();
        $dOutArray = $dOut->asArray();

        $batchSize = (int) (count($dOutArray[0]) / $this->outputLength);

        // Initialize gradient with zeros
        $dInput = array_fill(0, $inputChannels, array_fill(0, $inputLength * $batchSize, 0.0));

        // Distribute gradients to max positions
        foreach ($maxIndices as $c => $indices) {
            $outputPos = 0;

            foreach ($indices as $idx) {
                $dInput[$c][$idx] += $dOutArray[$c][$outputPos];
                ++$outputPos;
            }
        }

        return Matrix::quick($dInput);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "MaxPool1D (pool size: {$this->poolSize}, input length: {$this->inputLength},"
            . " stride: {$this->stride})";
    }
}
