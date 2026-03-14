<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\Helpers\Params;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

/**
 * Conv1D
 *
 * A 1-dimensional convolutional layer that applies sliding filters (kernels) over
 * sequential input data. It is useful for processing time series, signals, text
 * sequences, and other 1D data. Each filter slides across the input sequence and
 * computes dot products at each position, producing an output feature map.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Boorinio
 */
class Conv1D implements Hidden, Parametric
{
    /**
     * The number of output filters (output channels).
     *
     * @var positive-int
     */
    protected int $filters;

    /**
     * The size of the convolution kernel.
     *
     * @var positive-int
     */
    protected int $kernelSize;

    /**
     * The length of the input sequence.
     *
     * @var positive-int
     */
    protected int $inputLength;

    /**
     * The number of input channels.
     *
     * @var positive-int
     */
    protected int $inputChannels;

    /**
     * The stride of the convolution.
     *
     * @var positive-int
     */
    protected int $stride;

    /**
     * The amount of zero-padding to apply on both sides of the input.
     *
     * @var int<0,max>
     */
    protected int $padding;

    /**
     * The amount of L2 regularization applied to the weights.
     *
     * @var float
     */
    protected float $l2Penalty;

    /**
     * Should the layer include a bias parameter?
     *
     * @var bool
     */
    protected bool $bias;

    /**
     * The kernel weight initializer.
     *
     * @var Initializer
     */
    protected Initializer $kernelInitializer;

    /**
     * The bias initializer.
     *
     * @var Initializer
     */
    protected Initializer $biasInitializer;

    /**
     * The kernel weights.
     *
     * @var Parameter|null
     */
    protected ?Parameter $weights = null;

    /**
     * The biases.
     *
     * @var Parameter|null
     */
    protected ?Parameter $biases = null;

    /**
     * The computed output length.
     *
     * @var positive-int
     */
    protected int $outputLength;

    /**
     * The memorized inputs to the layer.
     *
     * @var Matrix|null
     */
    protected ?Matrix $input = null;

    /**
     * The padded input (if padding was applied).
     *
     * @var Matrix|null
     */
    protected ?Matrix $paddedInput = null;

    /**
     * @param int $filters Number of output filters (output channels)
     * @param int $kernelSize Size of the 1D convolution kernel
     * @param int $inputLength Length of the input sequence
     * @param int $inputChannels Number of input channels (default 1)
     * @param int $stride Convolution stride (default 1)
     * @param int $padding Zero-padding on both sides (default 0)
     * @param float $l2Penalty L2 regularization (default 0.0)
     * @param bool $bias Include bias parameter (default true)
     * @param Initializer|null $kernelInitializer Weight initializer
     * @param Initializer|null $biasInitializer Bias initializer
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $filters,
        int $kernelSize,
        int $inputLength,
        int $inputChannels = 1,
        int $stride = 1,
        int $padding = 0,
        float $l2Penalty = 0.0,
        bool $bias = true,
        ?Initializer $kernelInitializer = null,
        ?Initializer $biasInitializer = null
    ) {
        if ($filters < 1) {
            throw new InvalidArgumentException('Number of filters'
                . " must be greater than 0, $filters given.");
        }

        if ($kernelSize < 1) {
            throw new InvalidArgumentException('Kernel size must be'
                . " greater than 0, $kernelSize given.");
        }

        if ($inputLength < 1) {
            throw new InvalidArgumentException('Input length must be'
                . " greater than 0, $inputLength given.");
        }

        if ($inputChannels < 1) {
            throw new InvalidArgumentException('Number of input channels'
                . " must be greater than 0, $inputChannels given.");
        }

        if ($stride < 1) {
            throw new InvalidArgumentException('Stride must be'
                . " greater than 0, $stride given.");
        }

        if ($padding < 0) {
            throw new InvalidArgumentException('Padding cannot be'
                . " negative, $padding given.");
        }

        if ($l2Penalty < 0.0) {
            throw new InvalidArgumentException('L2 Penalty must be'
                . " greater than or equal to 0, $l2Penalty given.");
        }

        $outputLength = (int) floor(($inputLength + 2 * $padding - $kernelSize) / $stride) + 1;

        if ($outputLength < 1) {
            throw new InvalidArgumentException('Output length must be'
                . " greater than 0, $outputLength given. Check kernel size, stride, and padding values.");
        }

        $this->filters = $filters;
        $this->kernelSize = $kernelSize;
        $this->inputLength = $inputLength;
        $this->inputChannels = $inputChannels;
        $this->stride = $stride;
        $this->padding = $padding;
        $this->l2Penalty = $l2Penalty;
        $this->bias = $bias;
        $this->kernelInitializer = $kernelInitializer ?? new He();
        $this->biasInitializer = $biasInitializer ?? new Constant(0.0);
        $this->outputLength = $outputLength;
    }

    /**
     * Return the width of the layer (number of filters).
     *
     * @internal
     *
     * @return positive-int
     */
    public function width() : int
    {
        return $this->filters;
    }

    /**
     * Return the output length after convolution.
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
     * Return the kernel weight matrix.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return Matrix
     */
    public function weights() : Matrix
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        return $this->weights->param();
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @internal
     *
     * @param positive-int $fanIn (The fan in is not used; fan-in is calculated from inputChannels * kernelSize)
     * @return positive-int
     */
    public function initialize(int $fanIn) : int
    {
        $fanOut = $this->filters * $this->outputLength;

        // Initialize kernel weights: shape (filters, inputChannels * kernelSize)
        $kernelFanIn = $this->inputChannels * $this->kernelSize;
        $weights = $this->kernelInitializer->initialize($kernelFanIn, $this->filters);

        $this->weights = new Parameter($weights);

        if ($this->bias) {
            $biases = $this->biasInitializer->initialize(1, $this->filters)->columnAsVector(0);

            $this->biases = new Parameter($biases);
        }

        return $fanOut;
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
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        // Input shape: (inputChannels, inputLength * batchSize)
        $inputChannels = $input->m();

        if ($inputChannels !== $this->inputChannels) {
            throw new RuntimeException('Input channels mismatch:'
                . " expected {$this->inputChannels}, got {$inputChannels}.");
        }

        // Store the original input for backprop
        $this->input = $input;

        // Apply padding if needed
        $paddedInput = $this->padInput($input);
        $this->paddedInput = $paddedInput;

        $paddedLength = $this->inputLength + 2 * $this->padding;
        $batchSize = (int) ($input->n() / $this->inputLength);

        // Get kernel weights: shape (filters, inputChannels * kernelSize)
        $kernel = $this->weights->param();

        // Compute output: shape (filters, outputLength * batchSize)
        $output = $this->computeConvolution($paddedInput, $kernel, $paddedLength, $batchSize);

        // Add bias if enabled
        if ($this->biases) {
            $output = $this->addBias($output, $batchSize);
        }

        return $output;
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
        if (!$this->weights) {
            throw new RuntimeException('Layer is not initialized');
        }

        $inputChannels = $input->m();

        if ($inputChannels !== $this->inputChannels) {
            throw new RuntimeException('Input channels mismatch:'
                . " expected {$this->inputChannels}, got {$inputChannels}.");
        }

        $paddedInput = $this->padInput($input);
        $paddedLength = $this->inputLength + 2 * $this->padding;
        $batchSize = (int) ($input->n() / $this->inputLength);

        $kernel = $this->weights->param();

        $output = $this->computeConvolution($paddedInput, $kernel, $paddedLength, $batchSize);

        if ($this->biases) {
            $output = $this->addBias($output, $batchSize);
        }

        return $output;
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
        if (!$this->weights) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->input || !$this->paddedInput) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $dOut = $prevGradient();

        $batchSize = (int) ($this->input->n() / $this->inputLength);
        $paddedLength = $this->inputLength + 2 * $this->padding;

        $kernel = $this->weights->param();

        // Compute kernel gradient
        $dKernel = $this->computeKernelGradient($this->paddedInput, $dOut, $paddedLength, $batchSize);

        // Apply L2 penalty if needed
        if ($this->l2Penalty) {
            $dKernel = $dKernel->add($kernel->multiply($this->l2Penalty));
        }

        $this->weights->update($dKernel, $optimizer);

        // Update biases if enabled
        if ($this->biases) {
            // Sum gradients across all output positions
            $dB = $dOut->sum();

            $this->biases->update($dB, $optimizer);
        }

        $paddedInput = $this->paddedInput;
        $input = $this->input;

        $this->input = null;
        $this->paddedInput = null;

        return new Deferred([$this, 'gradient'], [$kernel, $dOut, $paddedInput, $input, $batchSize, $paddedLength]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param Matrix $kernel
     * @param Matrix $dOut
     * @param Matrix $paddedInput
     * @param Matrix $input
     * @param int $batchSize
     * @param int $paddedLength
     * @return Matrix
     */
    public function gradient(
        Matrix $kernel,
        Matrix $dOut,
        Matrix $paddedInput,
        Matrix $input,
        int $batchSize,
        int $paddedLength
    ) : Matrix {
        // Compute input gradient using full convolution with flipped kernel
        $dPaddedInput = $this->computeInputGradient($kernel, $dOut, $paddedLength, $batchSize);

        // Remove padding from gradient if needed
        if ($this->padding > 0) {
            return $this->unpadGradient($dPaddedInput, $input);
        }

        return $dPaddedInput;
    }

    /**
     * Return the parameters of the layer.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return Generator<Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'weights' => $this->weights;

        if ($this->biases) {
            yield 'biases' => $this->biases;
        }
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @internal
     *
     * @param Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->weights = $parameters['weights'];
        $this->biases = $parameters['biases'] ?? null;
    }

    /**
     * Pad the input matrix with zeros on both sides of the sequence.
     *
     * @param Matrix $input
     * @return Matrix
     */
    protected function padInput(Matrix $input) : Matrix
    {
        if ($this->padding === 0) {
            return $input;
        }

        $batchSize = (int) ($input->n() / $this->inputLength);
        $paddedLength = $this->inputLength + 2 * $this->padding;

        $padded = [];

        foreach ($input->asArray() as $channel => $row) {
            $paddedRow = [];

            for ($b = 0; $b < $batchSize; ++$b) {
                // Add left padding (zeros)
                for ($p = 0; $p < $this->padding; ++$p) {
                    $paddedRow[] = 0.0;
                }

                // Add original sequence
                for ($t = 0; $t < $this->inputLength; ++$t) {
                    $paddedRow[] = $row[$b * $this->inputLength + $t];
                }

                // Add right padding (zeros)
                for ($p = 0; $p < $this->padding; ++$p) {
                    $paddedRow[] = 0.0;
                }
            }

            $padded[] = $paddedRow;
        }

        return Matrix::quick($padded);
    }

    /**
     * Compute the convolution operation.
     *
     * @param Matrix $paddedInput Padded input matrix
     * @param Matrix $kernel Kernel weights
     * @param int $paddedLength Length of padded sequence
     * @param int $batchSize Number of samples in batch
     * @return Matrix
     */
    protected function computeConvolution(
        Matrix $paddedInput,
        Matrix $kernel,
        int $paddedLength,
        int $batchSize
    ) : Matrix {
        $output = [];

        $kernelArray = $kernel->asArray();
        $inputArray = $paddedInput->asArray();

        for ($f = 0; $f < $this->filters; ++$f) {
            $outputRow = [];

            for ($b = 0; $b < $batchSize; ++$b) {
                $sampleOffset = $b * $paddedLength;

                for ($t = 0; $t < $this->outputLength; ++$t) {
                    $startPos = $t * $this->stride;
                    $sum = 0.0;

                    // Sum contributions across all input channels
                    for ($c = 0; $c < $this->inputChannels; ++$c) {
                        $kernelOffset = $c * $this->kernelSize;

                        for ($k = 0; $k < $this->kernelSize; ++$k) {
                            $inputPos = $startPos + $k;
                            $sum += $inputArray[$c][$sampleOffset + $inputPos]
                                * $kernelArray[$f][$kernelOffset + $k];
                        }
                    }

                    $outputRow[] = $sum;
                }
            }

            $output[] = $outputRow;
        }

        return Matrix::quick($output);
    }

    /**
     * Add bias to the output.
     *
     * @param Matrix $output
     * @param int $batchSize
     * @return Matrix
     */
    protected function addBias(Matrix $output, int $batchSize) : Matrix
    {
        $biases = $this->biases;

        if (!$biases) {
            return $output;
        }

        $biasArray = $biases->param()->asArray();
        $outputArray = $output->asArray();

        $biased = [];

        foreach ($outputArray as $f => $row) {
            $bias = $biasArray[$f];
            $biasedRow = [];

            foreach ($row as $value) {
                $biasedRow[] = $value + $bias;
            }

            $biased[] = $biasedRow;
        }

        return Matrix::quick($biased);
    }

    /**
     * Compute the kernel gradient.
     *
     * @param Matrix $paddedInput
     * @param Matrix $dOut
     * @param int $paddedLength
     * @param int $batchSize
     * @return Matrix
     */
    protected function computeKernelGradient(
        Matrix $paddedInput,
        Matrix $dOut,
        int $paddedLength,
        int $batchSize
    ) : Matrix {
        $dKernel = [];

        $inputArray = $paddedInput->asArray();
        $dOutArray = $dOut->asArray();

        for ($f = 0; $f < $this->filters; ++$f) {
            $kernelRow = [];

            for ($c = 0; $c < $this->inputChannels; ++$c) {
                for ($k = 0; $k < $this->kernelSize; ++$k) {
                    $grad = 0.0;

                    for ($b = 0; $b < $batchSize; ++$b) {
                        $sampleOffset = $b * $paddedLength;
                        $outputOffset = $b * $this->outputLength;

                        for ($t = 0; $t < $this->outputLength; ++$t) {
                            $inputPos = $t * $this->stride + $k;
                            $grad += $inputArray[$c][$sampleOffset + $inputPos]
                                * $dOutArray[$f][$outputOffset + $t];
                        }
                    }

                    $kernelRow[] = $grad;
                }
            }

            $dKernel[] = $kernelRow;
        }

        return Matrix::quick($dKernel);
    }

    /**
     * Compute the input gradient using transposed convolution.
     *
     * @param Matrix $kernel
     * @param Matrix $dOut
     * @param int $paddedLength
     * @param int $batchSize
     * @return Matrix
     */
    protected function computeInputGradient(
        Matrix $kernel,
        Matrix $dOut,
        int $paddedLength,
        int $batchSize
    ) : Matrix {
        $dInput = [];

        $kernelArray = $kernel->asArray();
        $dOutArray = $dOut->asArray();

        // Flip kernel for transposed convolution
        $flippedKernel = [];

        foreach ($kernelArray as $f => $row) {
            $flippedKernel[$f] = array_reverse($row);
        }

        for ($c = 0; $c < $this->inputChannels; ++$c) {
            $inputRow = [];

            for ($b = 0; $b < $batchSize; ++$b) {
                $sampleOffset = $b * $paddedLength;
                $outputOffset = $b * $this->outputLength;

                for ($t = 0; $t < $paddedLength; ++$t) {
                    $grad = 0.0;

                    for ($f = 0; $f < $this->filters; ++$f) {
                        for ($k = 0; $k < $this->kernelSize; ++$k) {
                            // Calculate which output position contributes to this input position
                            $outT = ($t - $this->kernelSize + 1 + $k) / $this->stride;

                            if ($outT >= 0 && $outT < $this->outputLength && $outT == (int) $outT) {
                                $outT = (int) $outT;
                                $kernelOffset = $c * $this->kernelSize + $k;
                                $grad += $flippedKernel[$f][$kernelOffset]
                                    * $dOutArray[$f][$outputOffset + $outT];
                            }
                        }
                    }

                    $inputRow[] = $grad;
                }
            }

            $dInput[] = $inputRow;
        }

        return Matrix::quick($dInput);
    }

    /**
     * Remove padding from the gradient.
     *
     * @param Matrix $dPaddedInput
     * @param Matrix $originalInput
     * @return Matrix
     */
    protected function unpadGradient(Matrix $dPaddedInput, Matrix $originalInput) : Matrix
    {
        $dInput = [];
        $paddedArray = $dPaddedInput->asArray();

        foreach ($paddedArray as $c => $row) {
            $unpaddedRow = [];
            $batchSize = (int) ($originalInput->n() / $this->inputLength);
            $paddedLength = $this->inputLength + 2 * $this->padding;

            for ($b = 0; $b < $batchSize; ++$b) {
                $paddedOffset = $b * $paddedLength + $this->padding;

                for ($t = 0; $t < $this->inputLength; ++$t) {
                    $unpaddedRow[] = $row[$paddedOffset + $t];
                }
            }

            $dInput[] = $unpaddedRow;
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
        return "Conv1D (filters: {$this->filters}, kernel size: {$this->kernelSize},"
            . " input length: {$this->inputLength}, input channels: {$this->inputChannels},"
            . " stride: {$this->stride}, padding: {$this->padding},"
            . " l2 penalty: {$this->l2Penalty}, bias: " . Params::toString($this->bias) . ','
            . " kernel initializer: {$this->kernelInitializer},"
            . " bias initializer: {$this->biasInitializer})";
    }
}
