<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\Tensor\Vector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use RuntimeException;
use Generator;
use Closure;

/**
 * Batch Norm
 *
 * Normalize the activations of the previous layer such that the mean activation
 * is close to 0 and the activation standard deviation is close to 1. Batch Norm
 * can be used to reduce the amount of covariate shift within the network
 * making it possible to use higher learning rates and converge faster under
 * some circumstances.
 *
 * References:
 * [1] S. Ioffe et al. (2015). Batch Normalization: Accelerating Deep Network
 * Training by Reducing Internal Covariate Shift.
 * [2] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BatchNorm implements Hidden, Parametric
{
    /**
     * The initializer for the beta parameter.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $betaInitializer;

    /**
     * The initializer for the gamma parameter.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $gammaInitializer;

    /**
     * The width of the layer. i.e. the number of neurons.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The learnable centering parameter.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected $beta;

    /**
     * The learnable scaling parameter.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected $gamma;

    /**
     * The running mean of each input dimension.
     *
     * @var \Rubix\Tensor\Vector|null
     */
    protected $mean;

    /**
     * The running variance of each input dimension.
     *
     * @var \Rubix\Tensor\Vector|null
     */
    protected $variance;

    /**
     * The running standard deviation of each input dimension.
     *
     * @var \Rubix\Tensor\Vector|null
     */
    protected $stddev;

    /**
     * The number of training samples that have passed through the layer so far.
     *
     * @var int|null
     */
    protected $n;

    /**
     * A cache of inverse standard deviations calculated during the forward pass.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $stdInv;

    /**
     * A cache of normalized inputs to the layer.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $xHat;

    /**
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $betaInitializer
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $gammaInitializer
     */
    public function __construct(?Initializer $betaInitializer = null, ?Initializer $gammaInitializer = null)
    {
        $this->betaInitializer = $betaInitializer ?? new Constant(0.);
        $this->gammaInitializer = $gammaInitializer ?? new Constant(1.);
    }

    /**
     * Return the width of the layer.
     *
     * @return int|null
     */
    public function width() : ?int
    {
        return $this->width;
    }

    /**
     * Return the parameters of the layer.
     *
     * @throws \RuntimeException
     * @return \Generator
     */
    public function parameters() : Generator
    {
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initilaized.');
        }

        yield $this->beta;
        yield $this->gamma;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param int $fanIn
     * @return int
     */
    public function initialize(int $fanIn) : int
    {
        $fanOut = $fanIn;

        $this->mean = Vector::zeros($fanIn);
        $this->variance = Vector::ones($fanIn);

        $beta = $this->betaInitializer->initialize($fanIn, 1);
        $gamma = $this->gammaInitializer->initialize($fanIn, 1);

        $this->beta = new Parameter($beta);
        $this->gamma = new Parameter($gamma);

        $this->width = $fanOut;

        $this->n = 0;

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \Rubix\Tensor\Matrix $input
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($this->n === null or empty($this->mean) or empty($this->variance) or !$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $n = $input->n();

        $bHat = $this->beta->w()->row(0);
        $gHat = $this->gamma->w()->row(0);

        $oldMeans = $this->mean->asArray();
        $oldVariances = $this->variance->asArray();
        $oldWeight = $this->n;

        $newMeans = $newVars = $stddevs = $stdInv = $xHat = $out = [];

        foreach ($input as $i => $row) {
            [$mean, $variance] = Stats::meanVar($row);

            $oldMean = $oldMeans[$i];

            $newMeans[] = (($n * $mean)
                + ($oldWeight * $oldMean))
                / ($oldWeight + $n);

            $newVars[] = ($oldWeight
                * $oldVariances[$i] + ($n * $variance)
                + ($oldWeight / ($n * ($oldWeight + $n)))
                * ($n * $oldMean - $n * $mean) ** 2)
                / ($oldWeight + $n);

            $stddevs[] = $stddev = sqrt($variance ?: self::EPSILON);

            $beta = $bHat[$i];
            $gamma = $gHat[$i];

            $stdInvRow = $xHatRow = $outRow = [];

            foreach ($row as $value) {
                $alpha = 1. / $stddev;

                $tau = $alpha * ($value - $mean);

                $stdInvRow[] = $alpha;
                $xHatRow[] = $tau;
                $outRow[] = $gamma * $tau + $beta;
            }

            $stdInv[] = $stdInvRow;
            $xHat[] = $xHatRow;
            $out[] = $outRow;
        }

        $this->mean = Vector::quick($newMeans);
        $this->variance = Vector::quick($newVars);
        $this->stddev = Vector::quick($stddevs);
        $this->stdInv = Matrix::quick($stdInv);
        $this->xHat = Matrix::quick($xHat);

        $this->n += $n;

        return Matrix::quick($out);
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \Rubix\Tensor\Matrix $input
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if (!$this->mean or !$this->stddev or !$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initilaized.');
        }
        
        $bHat = $this->beta->w()->row(0);
        $gHat = $this->gamma->w()->row(0);

        $out = [];

        foreach ($input as $i => $row) {
            $mean = $this->mean[$i];
            $stddev = $this->stddev[$i];
            
            $beta = $bHat[$i];
            $gamma = $gHat[$i];

            $vector = [];

            foreach ($row as $value) {
                $vector[] = $gamma * ($value - $mean) / $stddev + $beta;
            }

            $out[] = $vector;
        }

        return Matrix::quick($out);
    }

    /**
     * Calculate the errors and gradients of the layer and update the parameters.
     *
     * @param Closure $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return Closure
     */
    public function back(Closure $prevGradient, Optimizer $optimizer) : Closure
    {
        if ($this->n === null or !$this->mean or !$this->stddev or !$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initilaized.');
        }

        if (!$this->stdInv or !$this->xHat) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradient();

        $dBeta = $dOut->sum()->asRowMatrix();
        $dGamma = $dOut->multiply($this->xHat)->sum()->asRowMatrix();

        $gamma = $this->gamma->w()->rowAsVector(0);

        $optimizer->step($this->beta, $dBeta);
        $optimizer->step($this->gamma, $dGamma);

        $stdInv = $this->stdInv;
        $xHat = $this->xHat;

        unset($this->stdInv, $this->xHat);

        return function () use ($dOut, $gamma, $stdInv, $xHat) {
            $dXHat = $dOut->multiply($gamma->transpose());

            $xHatSigma = $dXHat->multiply($xHat)->sum();

            $dXHatSigma = $dXHat->sum();

            return $dXHat->multiply($dOut->m())
                ->subtract($dXHatSigma)
                ->subtract($xHat->multiply($xHatSigma))
                ->multiply($stdInv->divide($dOut->m()));
        };
    }

    /**
     * Return the parameters of the layer in an associative array.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function read() : array
    {
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initilaized.');
        }

        return [
            'beta' => clone $this->beta,
            'gamma' => clone $this->gamma,
        ];
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param array $parameters
     * @throws \RuntimeException
     */
    public function restore(array $parameters) : void
    {
        $this->beta = $parameters['beta'];
        $this->gamma = $parameters['gamma'];
    }
}
