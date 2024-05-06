<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Tensor\ColumnVector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use const Rubix\ML\EPSILON;

/**
 * Batch Norm
 *
 * Normalize the activations of the previous layer such that the mean activation
 * is close to 0 and the standard deviation is close to 1. Batch Norm can reduce
 * the amount of covariate shift within the network which makes it possible to use
 * higher learning rates and converge faster under some circumstances.
 *
 * References:
 * [1] S. Ioffe et al. (2015). Batch Normalization: Accelerating Deep Network
 * Training by Reducing Internal Covariate Shift.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BatchNorm implements Hidden, Parametric
{
    /**
     * The decay rate of the previous running averages of the global mean and variance.
     *
     * @var float
     */
    protected float $decay;

    /**
     * The initializer for the beta parameter.
     *
     * @var Initializer
     */
    protected Initializer $betaInitializer;

    /**
     * The initializer for the gamma parameter.
     *
     * @var Initializer
     */
    protected Initializer $gammaInitializer;

    /**
     * The width of the layer. i.e. the number of neurons.
     *
     * @var positive-int|null
     */
    protected ?int $width = null;

    /**
     * The learnable centering parameter.
     *
     * @var Parameter|null
     */
    protected ?Parameter $beta = null;

    /**
     * The learnable scaling parameter.
     *
     * @var Parameter|null
     */
    protected ?Parameter $gamma = null;

    /**
     * The running mean of each input dimension.
     *
     * @var \Tensor\Vector|null
     */
    protected ?\Tensor\Vector $mean = null;

    /**
     * The running variance of each input dimension.
     *
     * @var \Tensor\Vector|null
     */
    protected ?\Tensor\Vector $variance = null;

    /**
     * A cache of inverse standard deviations calculated during the forward pass.
     *
     * @var \Tensor\Vector|null
     */
    protected ?\Tensor\Vector $stdInv = null;

    /**
     * A cache of normalized inputs to the layer.
     *
     * @var Matrix|null
     */
    protected ?Matrix $xHat = null;

    /**
     * @param float $decay
     * @param Initializer|null $betaInitializer
     * @param Initializer|null $gammaInitializer
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $decay = 0.1,
        ?Initializer $betaInitializer = null,
        ?Initializer $gammaInitializer = null
    ) {
        if ($decay < 0.0 or $decay > 1.0) {
            throw new InvalidArgumentException('Decay must be'
                . " between 0 and 1, $decay given.");
        }

        $this->decay = $decay;
        $this->betaInitializer = $betaInitializer ?? new Constant(0.0);
        $this->gammaInitializer = $gammaInitializer ?? new Constant(1.0);
    }

    /**
     * Return the width of the layer.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return positive-int
     */
    public function width() : int
    {
        if ($this->width === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        return $this->width;
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
        $fanOut = $fanIn;

        $beta = $this->betaInitializer->initialize(1, $fanOut)->columnAsVector(0);
        $gamma = $this->gammaInitializer->initialize(1, $fanOut)->columnAsVector(0);

        $this->beta = new Parameter($beta);
        $this->gamma = new Parameter($gamma);

        $this->width = $fanOut;

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
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $mean = $input->mean();
        $variance = $input->variance($mean)->clipLower(EPSILON);
        $stdInv = $variance->sqrt()->reciprocal();

        $xHat = $stdInv->multiply($input->subtract($mean));

        if (!$this->mean or !$this->variance) {
            $this->mean = $mean;
            $this->variance = $variance;
        }

        $this->mean = $this->mean->multiply(1.0 - $this->decay)
            ->add($mean->multiply($this->decay));

        $this->variance = $this->variance->multiply(1.0 - $this->decay)
            ->add($variance->multiply($this->decay));

        $this->stdInv = $stdInv;
        $this->xHat = $xHat;

        return $this->gamma->param()->multiply($xHat)
            ->add($this->beta->param());
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @internal
     *
     * @param Matrix $input
     * @throws RuntimeException
     * @return Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if (!$this->mean or !$this->variance or !$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $xHat = $input->subtract($this->mean)
            ->divide($this->variance->sqrt());

        return $this->gamma->param()->multiply($xHat)
            ->add($this->beta->param());
    }

    /**
     * Calculate the errors and gradients of the layer and update the parameters.
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
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->stdInv or !$this->xHat) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradient();

        $dBeta = $dOut->sum();
        $dGamma = $dOut->multiply($this->xHat)->sum();

        $gamma = $this->gamma->param();

        $this->beta->update($dBeta, $optimizer);
        $this->gamma->update($dGamma, $optimizer);

        $stdInv = $this->stdInv;
        $xHat = $this->xHat;

        $this->stdInv = $this->xHat = null;

        return new Deferred(
            [$this, 'gradient'],
            [$dOut, $gamma, $stdInv, $xHat]
        );
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @internal
     *
     * @param Matrix $dOut
     * @param ColumnVector $gamma
     * @param ColumnVector $stdInv
     * @param Matrix $xHat
     * @return Matrix
     */
    public function gradient(Matrix $dOut, ColumnVector $gamma, ColumnVector $stdInv, Matrix $xHat) : Matrix
    {
        $dXHat = $dOut->multiply($gamma);

        $xHatSigma = $dXHat->multiply($xHat)->sum();

        $dXHatSigma = $dXHat->sum();

        return $dXHat->multiply($dOut->m())
            ->subtract($dXHatSigma)
            ->subtract($xHat->multiply($xHatSigma))
            ->multiply($stdInv->divide($dOut->m()));
    }

    /**
     * Return the parameters of the layer.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return \Generator<\Rubix\ML\NeuralNet\Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'beta' => $this->beta;
        yield 'gamma' => $this->gamma;
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @internal
     *
     * @param \Rubix\ML\NeuralNet\Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->beta = $parameters['beta'];
        $this->gamma = $parameters['gamma'];
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
        return "Batch Norm (decay: {$this->decay}, beta initializer: {$this->betaInitializer},"
            . " gamma initializer: {$this->gammaInitializer})";
    }
}
